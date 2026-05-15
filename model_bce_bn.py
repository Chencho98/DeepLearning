# model_bce_bn.py — Conditional VAE para HAM10000
# Autocodificador Variacional Condicional (CVAE)
# Aquí se define el modelo principal.
# La idea es combinar:
# - compresión de la imagen,
# - representación latente probabilística,
# - y generación condicionada por clase.
#
# Este modelo sirve para reconstruir imágenes y también para generar nuevas
# imágenes de una clase específica de lesión.

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# Constantes del modelo
# =========================

# Usamos 128x128 porque es un tamaño bueno entre calidad visual y costo computacional.
# Más grande aumenta el costo, más pequeño puede perder detalles importantes de la lesión.
IMG_SIZE = 128

# El espacio latente de 128 dimensiones da suficiente capacidad
# para guardar información de la imagen sin hacerlo demasiado grande.
# Si fuera muy pequeño, el modelo perdería detalle;
# si fuera muy grande, el espacio latente sería más difícil de aprender.
LATENT_DIM = 128

# Número de clases del dataset HAM10000.
# Cada clase representa un tipo de lesión de piel.
NUM_CLASSES = 7

# ImageFolder ordena las carpetas alfabéticamente.
# Por eso guardamos el orden esperado de clases.
CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

# Descripción médica simple de cada clase.
# Esto ayuda a interpretar el dataset y entender qué representa cada etiqueta.
CLASS_DESCRIPTIONS = {
    "mel":   "Melanoma — cáncer agresivo de melanocitos",
    "nv":    "Nevi melanocíticos — lunares benignos",
    "bcc":   "Carcinoma basocelular — cáncer frecuente, lento",
    "akiec": "Queratosis actínica / Bowen — lesión precancerosa",
    "bkl":   "Queratosis benigna — manchas benignas",
    "df":    "Dermatofibroma — bulto benigno firme",
    "vasc":  "Lesiones vasculares — hemangiomas y similares",
}

# Estos límites se usan para evitar inestabilidad numérica.
# En un VAE es común que logvar, std o z crezcan demasiado y generen NaN o Inf.
LOGVAR_MIN = -6.0
LOGVAR_MAX = 2.0
STD_MAX = 3.0
Z_CLAMP = 10.0


# =========================
# Inicialización de pesos
# =========================

# La inicialización Xavier ayuda a que las capas empiecen con valores estables.
# Esto mejora la convergencia al comienzo del entrenamiento.
def init_weights(module):
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

    # BatchNorm empieza de forma neutra:
    # peso en 1 y sesgo en 0.
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


# =========================
# Bloque residual simple
# =========================

# Este bloque residual sirve para conservar información mientras la red aprende.
# La idea es que la salida no dependa solo de las convoluciones nuevas,
# sino también de la entrada original.
#
# Esto ayuda a:
# - mejorar el flujo de gradiente,
# - hacer la red más estable,
# - y evitar que se pierda información importante.
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        # Se suma la entrada con la salida del bloque.
        # Esa suma residual ayuda a que la red no pierda la señal original.
        return F.relu(x + self.block(x), inplace=True)


# =========================
# Encoder
# =========================

# El encoder se encarga de comprimir la imagen.
# Va reduciendo la resolución espacial y aumentando la cantidad de canales.
#
# Arquitectura:
# - 128x128 -> 64x64 -> 32x32 -> 16x16 -> 8x8
# - luego se aplana la representación
# - y se generan mu y logvar
#
# En un VAE no se guarda solo un vector fijo,
# sino una distribución en el espacio latente.
class Encoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()

        self.conv = nn.Sequential(
            # 128x128x3 -> 64x64x32
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            ResBlock(32),

            # 64x64x32 -> 32x32x64
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResBlock(64),

            # 32x32x64 -> 16x16x128
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            ResBlock(128),

            # 16x16x128 -> 8x8x256
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # Después de las convoluciones tenemos un mapa 8x8x256.
        # Eso se aplana antes de pasar a capas fully connected.
        self.flatten_dim = 256 * 8 * 8

        # El VAE necesita dos salidas:
        # - mu: centro de la distribución
        # - logvar: incertidumbre de esa distribución
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

        # Aplicamos la inicialización definida arriba.
        self.apply(init_weights)

        # Empezar con varianza pequeña ayuda a que el muestreo no sea caótico al inicio.
        nn.init.constant_(self.fc_logvar.bias, -2.0)

    def forward(self, x):
        # La imagen pasa por el extractor convolucional.
        h = self.conv(x)
        h = h.view(h.size(0), -1)

        # Sacamos la media de la distribución latente.
        mu = self.fc_mu(h)

        # Sacamos la log-varianza y la acotamos para estabilidad.
        logvar = self.fc_logvar(h)
        logvar = torch.clamp(logvar, min=LOGVAR_MIN, max=LOGVAR_MAX)

        return mu, logvar


# =========================
# Decoder
# =========================

# El decoder hace lo contrario al encoder:
# toma el vector latente y reconstruye la imagen.
#
# Además es condicional:
# recibe la clase de la lesión como embedding para guiar la generación.
# Eso hace que el modelo no genere imágenes "genéricas",
# sino imágenes asociadas a una clase concreta.
class Decoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, num_classes=NUM_CLASSES):
        super().__init__()

        # Convertimos la clase en un embedding denso.
        # Así la red no trabaja con la etiqueta como simple número.
        self.class_embed = nn.Embedding(num_classes, 32)

        # Unimos z + embedding de la clase.
        # Esa mezcla le da contexto semántico al decoder.
        self.fc = nn.Linear(latent_dim + 32, 256 * 8 * 8)

        self.deconv = nn.Sequential(
            # Mantenemos información inicial antes de empezar a subir resolución.
            ResBlock(256),

            # 8x8x256 -> 16x16x128
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            ResBlock(128),

            # 16x16x128 -> 32x32x64
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResBlock(64),

            # 32x32x64 -> 64x64x32
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            ResBlock(32),

            # 64x64x32 -> 128x128x3
            # La última capa devuelve logits, no sigmoid.
            # Eso permite usar BCEWithLogitsLoss de forma más estable.
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
        )

        self.apply(init_weights)

    def forward(self, z, class_label):
        # Convertimos la clase en embedding.
        ce = self.class_embed(class_label)

        # Concatenamos el vector latente con la clase.
        zc = torch.cat([z, ce], dim=1)

        # Expandimos a un mapa de características.
        h = self.fc(zc)
        h = h.view(h.size(0), 256, 8, 8)

        # Obtenemos la imagen reconstruida en forma de logits.
        logits = self.deconv(h)
        return logits


# =========================
# CVAE completo
# =========================

# Aquí se une todo el modelo:
# encoder + reparametrización + decoder.
#
# Esta clase representa el modelo final que se usa para entrenar,
# reconstruir, generar e interpolar imágenes.
class ConditionalVAE(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, num_classes=NUM_CLASSES, beta=1.0):
        super().__init__()

        self.latent_dim = latent_dim
        self.beta = beta

        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim, num_classes)

    # La reparametrización es la parte clave del VAE.
    # En vez de muestrear directamente de una distribución no diferenciable,
    # usamos:
    # z = mu + eps * std
    #
    # mu es la media del espacio latente.
    # std es la desviación estándar.
    # eps es un ruido aleatorio tomado de una distribución normal estándar, normalmente N(0,1).
    # z es el vector latente final que se manda al decoder.
    #
    # Así el muestreo se vuelve entrenable con backpropagation.
    def reparametrize(self, mu, logvar, sample=True):
        # En validación podemos usar mu directo para una reconstrucción determinista.
        if not sample:
            return torch.clamp(mu, -Z_CLAMP, Z_CLAMP)

        # Convertimos logvar a desviación estándar.
        std = torch.exp(0.5 * logvar)
        std = torch.clamp(std, max=STD_MAX)

        # Ruido gaussiano.
        eps = torch.randn_like(std)

        # Muestreo reparametrizado.
        z = mu + eps * std
        z = torch.clamp(z, -Z_CLAMP, Z_CLAMP)

        return z

    def forward(self, x, label, sample=True):
        # El encoder extrae la distribución latente.
        mu, logvar = self.encoder(x)

        # Muestreamos o usamos mu directo según el modo.
        z = self.reparametrize(mu, logvar, sample=sample)

        # El decoder reconstruye usando z y la clase.
        x_logits = self.decoder(z, label)

        return x_logits, mu, logvar

    # La función de pérdida combina dos ideas:
    #
    # 1) Reconstrucción:
    #    hace que la imagen generada se parezca a la original.
    #
    # 2) KL Divergence:
    #    obliga al espacio latente a parecerse a una distribución normal.
    #
    # Esa combinación es la base de un VAE.
    def loss(self, x, x_logits, mu, logvar, beta=None):
        if beta is None:
            beta = self.beta

        B = x.size(0)

        x_f = x.float()
        logits_f = x_logits.float()
        mu_f = mu.float()
        logvar_f = logvar.float()

        # BCEWithLogitsLoss se usa porque el decoder devuelve logits.
        # Esto es más estable que aplicar sigmoid primero y luego BCE.
        #
        # Además las imágenes deben estar en [0, 1],
        # por eso en el entrenamiento no usamos Normalize().
        recon_loss = F.binary_cross_entropy_with_logits(
            logits_f,
            x_f,
            reduction="sum",
        ) / B

        # KL divergence:
        # mide qué tan lejos está la distribución latente de una normal.
        # Esto hace que el espacio latente sea más ordenado y generativo.
        kl_loss = 0.5 * torch.sum(
            logvar_f.exp() + mu_f.pow(2) - 1.0 - logvar_f
        ) / B

        total = recon_loss + beta * kl_loss

        return total, recon_loss, kl_loss

    @torch.no_grad()
    def generate(self, class_label, n=1, device="cpu", temperature=1.0):
        # Generación de imágenes nuevas.
        # Se muestrea un z aleatorio y se condiciona por la clase.
        self.eval()

        label = torch.full(
            size=(n,),
            fill_value=int(class_label),
            dtype=torch.long,
            device=device,
        )

        # Temperature controla qué tan dispersa es la generación.
        # Valores más altos dan más variedad, pero pueden hacer la imagen menos estable.
        z = torch.randn(n, self.latent_dim, device=device) * temperature
        z = torch.clamp(z, -Z_CLAMP, Z_CLAMP)

        logits = self.decoder(z, label)

        # Aplicamos sigmoid solo al final para convertir logits a [0, 1].
        return torch.sigmoid(logits)

    @torch.no_grad()
    def reconstruct(self, x, label):
        # Reconstrucción determinista.
        # Aquí usamos mu directo para ver la salida más estable del modelo.
        self.eval()

        mu, logvar = self.encoder(x)
        z = self.reparametrize(mu, logvar, sample=False)

        logits = self.decoder(z, label)
        return torch.sigmoid(logits), mu

    @torch.no_grad()
    def interpolate(self, x1, label1, x2, label2, steps=8, device="cpu"):
        # Interpolación en el espacio latente.
        # Sirve para ver si el modelo aprendió un espacio continuo.
        self.eval()

        mu1, _ = self.encoder(x1)
        mu2, _ = self.encoder(x2)

        alphas = torch.linspace(0, 1, steps, device=device)

        results = []

        for a in alphas:
            # Mezclamos los dos latentes para crear transiciones suaves.
            z = (1.0 - a) * mu1 + a * mu2
            z = torch.clamp(z, -Z_CLAMP, Z_CLAMP)

            # Se decodifica usando la clase del primer ejemplo.
            logits = self.decoder(z, label1)
            results.append(torch.sigmoid(logits))

        return torch.cat(results, dim=0)


# =========================
# Prueba rápida
# =========================

# Esta sección sirve como prueba mínima para verificar que:
# - el forward funciona,
# - la loss es finita,
# - la generación produce tensores válidos,
# - y el modelo no tiene errores de forma.
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ConditionalVAE(latent_dim=128, num_classes=7, beta=1.0).to(device)

    # Datos de prueba aleatorios.
    # Se usa este tamaño porque coincide con el formato real esperado del dataset.
    x = torch.rand(4, 3, 128, 128, device=device)
    label = torch.randint(0, 7, (4,), device=device)

    x_logits, mu, logvar = model(x, label, sample=True)
    loss, recon, kl = model.loss(x, x_logits, mu, logvar)

    print(f"Input:    {x.shape}  min={x.min():.3f}  max={x.max():.3f}")
    print(f"Logits:   {x_logits.shape}  min={x_logits.min():.3f}  max={x_logits.max():.3f}")
    print(f"mu:       min={mu.min():.3f}  max={mu.max():.3f}")
    print(f"logvar:   min={logvar.min():.3f}  max={logvar.max():.3f}")
    print(f"Loss:     total={loss:.4f}  recon={recon:.4f}  kl={kl:.4f}")

    assert torch.isfinite(loss), "NaN/Inf en loss"
    assert torch.isfinite(x_logits).all(), "NaN/Inf en logits"
    assert torch.isfinite(mu).all(), "NaN/Inf en mu"
    assert torch.isfinite(logvar).all(), "NaN/Inf en logvar"

    gen = model.generate(class_label=0, n=4, device=device)
    print(f"Generated: {gen.shape}  min={gen.min():.3f}  max={gen.max():.3f}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parámetros: {total_params:,}")
    print("✅ Test pasado")
