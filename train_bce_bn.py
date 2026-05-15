# train_bce_bn.py — Entrenamiento para model_bce_bn.py
# Aquí se define todo el flujo de entrenamiento del CVAE.
#
# Este archivo no solo entrena la red:
# también prepara los datos, controla estabilidad numérica,
# guarda checkpoints, evalúa en validación y genera muestras.
#
# La idea del diseño es que el entrenamiento sea:
# - estable,
# - reproducible,
# - fácil de reanudar,
# - y fácil de monitorear.

import os
import time
import shutil
import argparse
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image

from model_bce_bn import ConditionalVAE, IMG_SIZE, LATENT_DIM


# =========================
# Argumentos
# =========================

# Aquí se definen los parámetros que controlan todo el entrenamiento.
# Se usan valores por defecto razonables para que el script funcione sin cambiar mucho,
# pero se pueden ajustar según el hardware o el experimento.
def parse_args():
    p = argparse.ArgumentParser()

    # Carpetas del dataset
    p.add_argument("--train_dir", default="./balanced_image/training")
    p.add_argument("--val_dir", default="./balanced_image/testing")

    # Salida principal
    p.add_argument(
        "--output_dir",
        default="./output_cvae_bce_bn",
        help="Directorio nuevo para no sobreescribir resultados anteriores.",
    )

    # Carpeta opcional de Drive para backups
    p.add_argument(
        "--drive_dir",
        default=None,
        help="Carpeta opcional en Drive para backups.",
    )

    # Epochs:
    # Se usa 50 como punto de partida porque suele ser suficiente
    # En datasets más difíciles esto puede subir.
    p.add_argument("--epochs", type=int, default=50)

    # Batch size:
    # 64 es un valor intermedio bueno para equilibrar estabilidad y uso de memoria.
    p.add_argument("--batch_size", type=int, default=64)

    # Learning rate:
    # 1e-4 es una elección conservadora y estable para Adam.
    p.add_argument("--lr", type=float, default=1e-4)

    # Tamaño del latente y beta del VAE
    p.add_argument("--latent_dim", type=int, default=LATENT_DIM)
    p.add_argument("--beta", type=float, default=1.0)

    # Warmup de KL:
    # Se deja en 0 por defecto para empezar simple,
    # pero se puede activar si al inicio el KL domina demasiado.
    p.add_argument("--kl_warmup_epochs", type=int, default=0)

    # Tamaño de imagen y workers
    p.add_argument("--img_size", type=int, default=IMG_SIZE)
    p.add_argument("--num_workers", type=int, default=4)

    # Reanudar entrenamiento
    p.add_argument("--resume", default=None)
    p.add_argument("--auto_resume", action="store_true")

    # Cada cuántas épocas guardar checkpoint numerado
    p.add_argument("--save_every", type=int, default=5)

    # Opciones para acelerar o depurar
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--debug", action="store_true")

    return p.parse_args()


# =========================
# Helpers
# =========================

# Esta parte prepara el entorno para entrenamiento.
# Si debug está activo, se prioriza detectar errores antes que velocidad.
def setup_runtime(debug):
    if debug:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        torch.autograd.set_detect_anomaly(True)
        cudnn.deterministic = True
        cudnn.benchmark = False
        print("⚠️  DEBUG activo: AMP desactivado, anomaly detection activo.\n")
    else:
        cudnn.benchmark = True


# Sirve para ver cuánta VRAM se está usando.
# Útil cuando el entrenamiento empieza a acercarse al límite de memoria.
def log_vram(device):
    if device.type == "cuda":
        alloc = torch.cuda.memory_allocated(device) / 1024**2
        reserved = torch.cuda.memory_reserved(device) / 1024**2
        print(f"  [DEBUG] VRAM: {alloc:.0f} MB alloc / {reserved:.0f} MB reserved")


# Imprime estadísticas básicas de un tensor.
# Esto ayuda a detectar NaN, Inf o valores fuera de rango.
def print_tensor_stats(name, x):
    with torch.no_grad():
        print(
            f"    {name:<8} finite={torch.isfinite(x).all().item()} "
            f"min={x.min().item():.6f} "
            f"max={x.max().item():.6f} "
            f"mean={x.mean().item():.6f}"
        )


# Warmup de KL:
# al comienzo del entrenamiento el KL puede ser demasiado fuerte.
# Subir beta poco a poco permite que primero aprenda a reconstruir,
# y después a regularizar el espacio latente.
def get_beta_for_epoch(epoch, target_beta, warmup_epochs):
    if warmup_epochs <= 0:
        return target_beta

    progress = min(1.0, epoch / float(warmup_epochs))
    return target_beta * progress


# Esta función ayuda a acceder al modelo real cuando está envuelto
# por DataParallel, compile o alguna otra capa.
def _raw(model):
    return getattr(model, "_orig_mod", model)


# =========================
# Dataset
# =========================

# Dataset personalizado para imágenes de piel.
# La idea es:
# - leer carpetas por clase,
# - filtrar clases no deseadas,
# - aplicar augmentations,
# - y dejar las imágenes en [0, 1] sin Normalize().
#
# No usamos Normalize porque el decoder y la loss trabajan con
# Binary Cross Entropy with Logits (BCEWithLogitsLoss) para clasificacion binaria,
# que espera targets entre 0 y 1.

#  Estructura esperada:
#
#    root_dir/
#      akiec/
#      bcc/
#      bkl/
#      df/
#      mel/
#      nv/
#      vasc/
#      normal_skin/   opcional, se excluye
#
#  IMPORTANTE:
#    No usamos Normalize().
#    Las imágenes quedan en [0, 1].
class SkinDataset(Dataset):
    def __init__(self, root_dir, img_size=128, augment=True, exclude_classes=None):
        if exclude_classes is None:
            exclude_classes = []

        if not os.path.isdir(root_dir):
            raise FileNotFoundError(f"No existe el directorio: {root_dir}")

        aug_transforms = []

        # Augmentation:
        # aumenta la variedad de imágenes sin cambiar la clase real.
        # Esto ayuda a generalizar mejor y reduce sobreajuste.
        if augment:
            aug_transforms = [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(
                    brightness=0.20,
                    contrast=0.20,
                    saturation=0.10,
                    hue=0.02,
                ),
            ]

        self.transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                *aug_transforms,
                transforms.ToTensor(),
            ]
        )

        # ImageFolder lee el dataset por carpetas.
        full_dataset = ImageFolder(root=root_dir)

        # Filtramos clases no deseadas, por ejemplo normal_skin.
        filtered_samples = [
            (path, label)
            for path, label in full_dataset.samples
            if full_dataset.classes[label] not in exclude_classes
        ]

        if len(filtered_samples) == 0:
            raise RuntimeError(f"No se encontraron imágenes válidas en {root_dir}")

        # Nos quedamos solo con las clases que realmente existen después del filtro.
        kept_classes = sorted(
            set(full_dataset.classes[label] for _, label in filtered_samples)
        )

        # Reindexamos clases para que train y val tengan el mismo mapeo.
        self.class_to_idx = {cls: i for i, cls in enumerate(kept_classes)}
        self.idx_to_class = {i: cls for cls, i in self.class_to_idx.items()}

        self.samples = [
            (path, self.class_to_idx[full_dataset.classes[label]])
            for path, label in filtered_samples
        ]

        self.targets = [label for _, label in self.samples]

        # Contamos imágenes por clase.
        class_counts = np.bincount(self.targets, minlength=len(kept_classes))

        if np.any(class_counts == 0):
            raise RuntimeError(f"Hay clases sin imágenes: {class_counts}")

        # WeightedRandomSampler usa pesos inversos a la frecuencia.
        # Esto ayuda cuando el dataset está desbalanceado.
        weights = 1.0 / class_counts
        self.sample_weights = torch.DoubleTensor(weights[self.targets])

        print(f"\nDataset: {root_dir}")
        print(f"Clases ({len(kept_classes)}): {self.class_to_idx}")
        print(f"Total: {len(self.samples)} imágenes")

        for cls, idx in self.class_to_idx.items():
            print(f"  {cls:<10} {class_counts[idx]:>6}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Error leyendo imagen {path}: {e}")

        img = self.transform(img)
        label = torch.tensor(label, dtype=torch.long)

        return img, label


# =========================
# DataLoader
# =========================

# Creamos el loader con opciones pensadas para rendimiento.
# pin_memory ayuda cuando se usa GPU.
# persistent_workers y prefetch_factor pueden acelerar la carga de datos.
def make_loader(dataset, batch_size, num_workers, use_amp, sampler=None, shuffle=False):
    kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": use_amp,
        "shuffle": shuffle if sampler is None else False,
        "sampler": sampler,
    }

    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2

    return DataLoader(dataset, **kwargs)


# =========================
# Checkpoints
# =========================

# Guardamos el estado completo del entrenamiento:
# modelo, optimizador, scheduler y scaler.
# Esto permite reanudar sin perder progreso.
def save_checkpoint(
    path,
    epoch,
    model,
    optimizer,
    scheduler,
    scaler,
    best_val,
    class_names,
    args,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "model": _raw(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "best_val": best_val,
            "class_names": class_names,
            "args": vars(args),
        },
        path,
    )


# Carga un checkpoint y recupera el estado para seguir entrenando.
def load_checkpoint(path, model, optimizer, scheduler, scaler, device):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No existe checkpoint: {path}")

    ckpt = torch.load(path, map_location=device, weights_only=False)

    _raw(model).load_state_dict(ckpt["model"])

    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])

    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])

    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])

    start_epoch = int(ckpt["epoch"]) + 1
    best_val = float(ckpt.get("best_val", float("inf")))

    print(f"✅ Checkpoint cargado: {path}")
    print(f"   Siguiente epoch: {start_epoch}")
    print(f"   Mejor val: {best_val:.4f}")

    return start_epoch, best_val


# Backup opcional a Drive.
# Esto es útil si se está entrenando en un entorno temporal
# y no se quiere perder el trabajo.
def backup_to_drive(src, drive_dir, name):
    if not drive_dir:
        return

    try:
        os.makedirs(drive_dir, exist_ok=True)
        dst = os.path.join(drive_dir, name)
        shutil.copy2(src, dst)
        print(f"  → Drive backup: {dst}")
    except Exception as e:
        print(f"  ⚠️ Drive backup falló: {e}")


# =========================
# Train / Val
# =========================

# Una época de entrenamiento.
# Aquí se hace:
# - forward,
# - cálculo de pérdida,
# - backward,
# - clipping de gradientes,
# - y update del optimizador.
#
# Se usa gradient clipping porque en modelos generativos
# los gradientes pueden crecer demasiado y volver inestable el entrenamiento.
def train_epoch(model, loader, optimizer, device, scaler, use_amp, debug, beta):
    model.train()

    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    skipped = 0
    first = True

    for batch_idx, (imgs, labels) in enumerate(loader):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # BCEWithLogitsLoss necesita targets en [0, 1].
        imgs = torch.clamp(imgs, 0.0, 1.0)

        optimizer.zero_grad(set_to_none=True)

        # AMP acelera el entrenamiento en GPU y reduce uso de memoria.
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            x_logits, mu, logvar = model(imgs, labels, sample=True)
            loss, recon, kl = model.loss(imgs, x_logits, mu, logvar, beta=beta)

        # Si algo sale mal numéricamente, se evita hacer step.
        if not torch.isfinite(loss):
            skipped += 1
            print(f"\n⛔ Batch {batch_idx}: loss no finita. Se omite.")
            print_tensor_stats("imgs", imgs)
            print_tensor_stats("logits", x_logits)
            print_tensor_stats("mu", mu)
            print_tensor_stats("logvar", logvar)
            optimizer.zero_grad(set_to_none=True)
            continue

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            # Clipping para evitar explosión de gradientes.
            grad_norm = torch.nn.utils.clip_grad_norm_(
                _raw(model).parameters(),
                max_norm=1.0,
                error_if_nonfinite=False,
            )

            if not torch.isfinite(grad_norm):
                skipped += 1
                print(f"\n⛔ Batch {batch_idx}: grad_norm no finito. Se omite step.")
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                continue

            scaler.step(optimizer)
            scaler.update()

        else:
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                _raw(model).parameters(),
                max_norm=1.0,
                error_if_nonfinite=False,
            )

            if not torch.isfinite(grad_norm):
                skipped += 1
                print(f"\n⛔ Batch {batch_idx}: grad_norm no finito. Se omite step.")
                optimizer.zero_grad(set_to_none=True)
                continue

            optimizer.step()

        total_loss += loss.item()
        total_recon += recon.item()
        total_kl += kl.item()

        # Debug solo en el primer batch para no saturar la salida.
        if debug and first:
            print("    [DEBUG TRAIN]")
            print_tensor_stats("imgs", imgs)
            print_tensor_stats("logits", x_logits)
            print_tensor_stats("probs", torch.sigmoid(x_logits))
            print_tensor_stats("mu", mu)
            print_tensor_stats("logvar", logvar)
            print(
                f"    loss={loss.item():.4f} "
                f"recon={recon.item():.4f} "
                f"kl={kl.item():.4f} "
                f"beta={beta:.4f} "
                f"grad_norm={float(grad_norm):.4f}"
            )
            first = False

    n = max(1, len(loader) - skipped)

    return (
        total_loss / n,
        total_recon / n,
        total_kl / n,
        skipped,
    )


# Validación sin backprop.
# Aquí se usa sample=False para que la reconstrucción sea determinista,
# usando mu directo en lugar de muestreo aleatorio.
@torch.no_grad()
def val_epoch(model, loader, device, use_amp, debug, beta):
    model.eval()

    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    skipped = 0
    first = True

    for batch_idx, (imgs, labels) in enumerate(loader):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        imgs = torch.clamp(imgs, 0.0, 1.0)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            x_logits, mu, logvar = model(imgs, labels, sample=False)
            loss, recon, kl = model.loss(imgs, x_logits, mu, logvar, beta=beta)

        if not torch.isfinite(loss):
            skipped += 1
            print(f"\n⛔ VAL batch {batch_idx}: loss no finita. Se omite.")
            print_tensor_stats("imgs", imgs)
            print_tensor_stats("logits", x_logits)
            print_tensor_stats("mu", mu)
            print_tensor_stats("logvar", logvar)
            continue

        total_loss += loss.item()
        total_recon += recon.item()
        total_kl += kl.item()

        if debug and first:
            print("    [DEBUG VAL]")
            print_tensor_stats("imgs", imgs)
            print_tensor_stats("logits", x_logits)
            print_tensor_stats("probs", torch.sigmoid(x_logits))
            print_tensor_stats("mu", mu)
            print_tensor_stats("logvar", logvar)
            print(
                f"    val_loss={loss.item():.4f} "
                f"val_recon={recon.item():.4f} "
                f"val_kl={kl.item():.4f} "
                f"beta={beta:.4f}"
            )
            first = False

    n = max(1, len(loader) - skipped)

    return (
        total_loss / n,
        total_recon / n,
        total_kl / n,
        skipped,
    )


# =========================
# Muestras
# =========================

# Generar muestras durante el entrenamiento permite revisar
# si el modelo está aprendiendo formas reales por clase.
# Guardamos una versión normal y otra normalizada para inspección visual.
@torch.no_grad()
def save_samples(model, class_names, device, output_dir, epoch, n_per_class=4):
    import torchvision.utils as vutils

    os.makedirs(output_dir, exist_ok=True)

    model.eval()

    all_imgs = []

    # Generamos varias imágenes por clase para comparar cómo cambia la salida.
    for class_idx in range(len(class_names)):
        imgs = model.generate(
            class_label=class_idx,
            n=n_per_class,
            device=device,
            temperature=0.8,
        )

        all_imgs.append(imgs.cpu())

    imgs = torch.cat(all_imgs, dim=0)

    # Imagen normal: respeta el rango [0, 1].
    path = os.path.join(output_dir, f"samples_epoch_{epoch:03d}.png")
    vutils.save_image(
        imgs,
        path,
        nrow=n_per_class,
        padding=2,
    )

    # Imagen normalizada: útil para ver contraste visual, aunque no sea la salida real.
    path_norm = os.path.join(output_dir, f"samples_epoch_{epoch:03d}_normalized.png")
    grid_norm = vutils.make_grid(
        imgs,
        nrow=n_per_class,
        normalize=True,
        padding=2,
    )
    vutils.save_image(grid_norm, path_norm)

    print(f"  → Muestras: {path}")
    print(f"  → Muestras normalizadas: {path_norm}")


# =========================
# Main
# =========================

# Aquí se junta todo:
# - se leen argumentos,
# - se preparan datos,
# - se crea el modelo,
# - se entrena época por época,
# - y se guardan resultados.
def main():
    args = parse_args()

    # Usamos GPU si está disponible.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    setup_runtime(args.debug)

    # AMP solo se usa en CUDA, y se desactiva si estamos en debug.
    use_amp = (device.type == "cuda") and (not args.no_amp) and (not args.debug)

    print(f"Dispositivo: {device} | AMP: {use_amp}")

    os.makedirs(args.output_dir, exist_ok=True)

    # =========================
    # Dataset
    # =========================

    # Train con augmentations para generalizar mejor.
    train_ds = SkinDataset(
        args.train_dir,
        img_size=args.img_size,
        augment=True,
        exclude_classes=["normal_skin"],
    )

    # Val sin augment para medir el comportamiento real del modelo.
    val_ds = SkinDataset(
        args.val_dir,
        img_size=args.img_size,
        augment=False,
        exclude_classes=["normal_skin"],
    )

    class_names = list(train_ds.class_to_idx.keys())
    num_classes = len(class_names)

    print(f"\nClases train ({num_classes}): {class_names}")
    print(f"Clases val: {list(val_ds.class_to_idx.keys())}")

    # Es importante que train y val usen exactamente el mismo mapeo de clases.
    if train_ds.class_to_idx != val_ds.class_to_idx:
        raise RuntimeError(
            "El mapeo de clases train/val no coincide. "
            f"train={train_ds.class_to_idx}, val={val_ds.class_to_idx}"
        )

    # Sampler balanceado para compensar clases con pocas imágenes.
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=train_ds.sample_weights,
        num_samples=len(train_ds),
        replacement=True,
    )

    train_loader = make_loader(
        train_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_amp=use_amp,
        sampler=sampler,
        shuffle=False,
    )

    val_loader = make_loader(
        val_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_amp=use_amp,
        sampler=None,
        shuffle=False,
    )

    # =========================
    # Modelo
    # =========================

    model = ConditionalVAE(
        latent_dim=args.latent_dim,
        num_classes=num_classes,
        beta=args.beta,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parámetros: {total_params:,}")

    # =========================
    # Optimizer
    # =========================

    # Adam suele funcionar bien en modelos generativos.
    # Se usa weight decay pequeño para regularización ligera.
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-5,
    )

    # ReduceLROnPlateau baja el learning rate si la validación deja de mejorar.
    # Esto ayuda a afinar el entrenamiento cuando el modelo se estanca.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=5,
        factor=0.5,
    )

    # GradScaler acompaña AMP para mantener estabilidad en precisión mixta.
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=use_amp,
        init_scale=1024,
        growth_interval=2000,
    )

    # =========================
    # Resume
    # =========================

    start_epoch = 1
    best_val = float("inf")

    latest_path = os.path.join(args.output_dir, "latest.pt")
    resume_path = args.resume

    # Si se activó auto_resume, usamos latest.pt si ya existe.
    if resume_path is None and args.auto_resume and os.path.exists(latest_path):
        resume_path = latest_path

    if resume_path:
        start_epoch, best_val = load_checkpoint(
            resume_path,
            model,
            optimizer,
            scheduler,
            scaler,
            device,
        )
    else:
        print("Iniciando desde cero.")

    # =========================
    # Logs
    # =========================

    # Guardamos las métricas en CSV para poder comparar experimentos después.
    log_path = os.path.join(args.output_dir, "training_log.csv")

    if not os.path.exists(log_path) or start_epoch == 1:
        with open(log_path, "w") as f:
            f.write(
                "epoch,beta,train_loss,train_recon,train_kl,"
                "val_loss,val_recon,val_kl,lr,elapsed_s,"
                "skipped_train,skipped_val\n"
            )

    best_path = os.path.join(args.output_dir, "best_model.pt")

    # =========================
    # Loop principal
    # =========================

    # Entrenamos por épocas para poder monitorear el avance,
    # guardar checkpoints y comparar train vs val.
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        # Si hay warmup, beta sube poco a poco.
        beta_now = get_beta_for_epoch(
            epoch=epoch,
            target_beta=args.beta,
            warmup_epochs=args.kl_warmup_epochs,
        )

        model.beta = beta_now

        if args.debug:
            print(f"\n── Epoch {epoch} TRAIN ──")

        tr_loss, tr_recon, tr_kl, skipped_train = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            use_amp=use_amp,
            debug=args.debug,
            beta=beta_now,
        )

        if args.debug:
            print(f"\n── Epoch {epoch} VAL ──")

        vl_loss, vl_recon, vl_kl, skipped_val = val_epoch(
            model=model,
            loader=val_loader,
            device=device,
            use_amp=use_amp,
            debug=args.debug,
            beta=beta_now,
        )

        # No castigamos el scheduler durante warmup.
        if epoch > args.kl_warmup_epochs:
            scheduler.step(vl_loss)

        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"beta={beta_now:.4f} | "
            f"Train={tr_loss:.4f} "
            f"(recon={tr_recon:.4f}, kl={tr_kl:.4f}) | "
            f"Val={vl_loss:.4f} "
            f"(recon={vl_recon:.4f}, kl={vl_kl:.4f}) | "
            f"LR={lr:.2e} | "
            f"{elapsed:.1f}s | "
            f"skip T/V={skipped_train}/{skipped_val}"
        )

        if args.debug:
            log_vram(device)

        # Guardamos el historial del entrenamiento.
        with open(log_path, "a") as f:
            f.write(
                f"{epoch},{beta_now:.6f},"
                f"{tr_loss:.6f},{tr_recon:.6f},{tr_kl:.6f},"
                f"{vl_loss:.6f},{vl_recon:.6f},{vl_kl:.6f},"
                f"{lr:.8f},{elapsed:.1f},"
                f"{skipped_train},{skipped_val}\n"
            )

        # Mejor modelo:
        # Si hay warmup, empezamos a evaluar el "mejor" después de eso.
        can_update_best = epoch >= max(1, args.kl_warmup_epochs)

        if can_update_best and vl_loss < best_val:
            best_val = vl_loss

            save_checkpoint(
                best_path,
                epoch,
                model,
                optimizer,
                scheduler,
                scaler,
                best_val,
                class_names,
                args,
            )

            print(f"  → Mejor modelo: {best_path} | val={best_val:.4f}")

        # latest se guarda siempre para poder retomar desde el último estado.
        save_checkpoint(
            latest_path,
            epoch,
            model,
            optimizer,
            scheduler,
            scaler,
            best_val,
            class_names,
            args,
        )

        # Cada cierto número de épocas guardamos un checkpoint extra
        # y generamos muestras para revisión visual.
        if epoch % args.save_every == 0:
            ckpt_name = f"ckpt_epoch_{epoch:03d}.pt"
            ckpt_path = os.path.join(args.output_dir, ckpt_name)

            save_checkpoint(
                ckpt_path,
                epoch,
                model,
                optimizer,
                scheduler,
                scaler,
                best_val,
                class_names,
                args,
            )

            print(f"  → Checkpoint: {ckpt_path}")

            # Generamos ejemplos visuales de cada clase.
            save_samples(
                model=model,
                class_names=class_names,
                device=device,
                output_dir=os.path.join(args.output_dir, "samples"),
                epoch=epoch,
                n_per_class=4,
            )

            # Si hay carpeta de Drive, copiamos también las muestras.
            if args.drive_dir:
                local_samples_dir = os.path.join(args.output_dir, "samples")
                drive_samples_dir = os.path.join(args.drive_dir, "samples")
                os.makedirs(drive_samples_dir, exist_ok=True)

                if os.path.isdir(local_samples_dir):
                    for fname in os.listdir(local_samples_dir):
                        if fname.endswith(".png"):
                            shutil.copy2(
                                os.path.join(local_samples_dir, fname),
                                os.path.join(drive_samples_dir, fname),
                            )

                    print(f"  → Samples backup en Drive: {drive_samples_dir}")
                else:
                    print(f"  ⚠ No existe carpeta local de samples: {local_samples_dir}")

            # Backups de checkpoint y logs.
            backup_to_drive(ckpt_path, args.drive_dir, ckpt_name)
            backup_to_drive(latest_path, args.drive_dir, "latest.pt")
            backup_to_drive(best_path, args.drive_dir, "best_model.pt")
            backup_to_drive(log_path, args.drive_dir, "training_log.csv")

    # Backup final al terminar.
    backup_to_drive(best_path, args.drive_dir, "best_model.pt")
    backup_to_drive(latest_path, args.drive_dir, "latest.pt")
    backup_to_drive(log_path, args.drive_dir, "training_log.csv")

    print(f"\n✅ Entrenamiento completo. Mejor val={best_val:.4f}")
    print(f"Resultados en: {args.output_dir}")


if __name__ == "__main__":
    main()
