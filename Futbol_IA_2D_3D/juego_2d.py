"""Fútbol 2D de estilo broadcast: partido local, cooperativo e IA."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, cast

import pygame

# ===== IA, datos, selecciones, audio y recursos integrados =====

from datetime import datetime
from pathlib import Path
import uuid
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
try:
    import keras
except Exception as error:  # TensorFlow puede fallar al cargar DLL en algunos equipos.
    keras = None
    print(f"Aviso: Keras no disponible ({type(error).__name__}); se usará movimiento de respaldo.")

MODOS = {1:"Humano vs PC", 2:"Humano vs Humano", 3:"1 IA vs 2 humanos", 4:"2 IA vs 2 humanos"}

def clamp(x, lo, hi):
    return float(np.clip(x, lo, hi))

class IAFutbol:
    def __init__(self):
        rng = np.random.default_rng(7)
        rows, labels = [], []
        for _ in range(3000):
            bx,bz,ax,az,tx,tz = rng.uniform(-1,1),rng.uniform(-.9,.9),rng.uniform(-1,1),rng.uniform(-.9,.9),rng.uniform(-1,1),rng.uniform(-.9,.9)
            da,dc,dg = np.hypot(bx-ax,bz-az),np.hypot(bx-tx,bz-tz),np.hypot(1-bx,bz)
            label = "apoyo" if dc+.08<da else "tiro" if dg<.34 and da<.24 else "pase" if da<.19 and dc<.4 and tx>bx+.18 else "defender" if bx<-.2 and da>.2 else "perseguir"
            rows.append(self.features(bx,bz,0,ax,az,tx,tz,dc)); labels.append(label)
        self.policy = make_pipeline(StandardScaler(), LogisticRegression(max_iter=300,class_weight="balanced")).fit(rows,labels)
        bx,yy,bz = rng.uniform(-1,1,3000),rng.uniform(0,1,3000),rng.uniform(-1,1,3000)
        target = np.column_stack([np.clip(.78*bx+np.where(bx>=0,.12,-.10)+.035*yy,-.85,.85),np.clip(bz*(.78-.1*yy),-.8,.8)])
        self.net = None
        if keras is not None:
            try:
                self.net = keras.Sequential([
                    keras.layers.Input((3,)),
                    keras.layers.Dense(20, activation="relu"),
                    keras.layers.Dense(12, activation="relu"),
                    keras.layers.Dense(2, activation="tanh"),
                ])
                self.net.compile(optimizer="adam", loss="mse")
                self.net.fit(np.column_stack([bx, yy, bz]), target,
                             epochs=4, batch_size=64, verbose=cast(Any, 0))
            except Exception as error:
                self.net = None
                print(f"Aviso: no se pudo entrenar Keras ({type(error).__name__}); se usará movimiento de respaldo.")
        self.log = MatchLog()

    @staticmethod
    def features(bx,bz,y,ax,az,tx,tz,dc):
        return [bx,bz,y,ax,az,tx,tz,np.hypot(bx-ax,bz-az),dc,np.hypot(1-bx,bz)]

    def decide(self, ball, actor, mate, sign):
        bx,bz = ball[0]*sign,ball[1]; ax,az = actor[0]*sign,actor[1]
        if mate is None: tx,tz,dc=ax,az,3.0
        else:
            tx,tz=mate[0]*sign,mate[1]; dc=float(np.hypot(bx-tx,bz-tz))
        action=str(self.policy.predict([self.features(bx,bz,ball[2],ax,az,tx,tz,dc)])[0])
        # La llamada directa evita la sobrecarga de Model.predict() en el bucle
        # del juego y mantiene la inferencia de Keras ligera por frame.
        if self.net is not None:
            pred = self.net(
                np.asarray([[bx, ball[2], bz]], dtype=np.float32), training=False
            ).numpy()[0]
        else:
            # Aproxima el objetivo con la misma regla usada para generar los
            # datos sintéticos de entrenamiento de Keras.
            pred = np.asarray([
                np.clip(.78 * bx + (.12 if bx >= 0 else -.10) + .035 * ball[2], -.85, .85),
                np.clip(bz * (.78 - .1 * ball[2]), -.8, .8),
            ], dtype=np.float32)
        return action,(clamp(pred[0]*sign,-.9,.9),clamp(pred[1],-.85,.85))

class MatchLog:
    def __init__(self):
        self.folder=Path(__file__).resolve().parent/"data"; self.folder.mkdir(exist_ok=True)
        self.matches=self.read("partidos.csv",["id","fecha","modo","goles_0","goles_1","resultado"])
        self.events=self.read("eventos.csv",["id","segundo","equipo","pais","jugador","evento"])
        self.players=self.read("jugadores.csv",["id","equipo","pais","jugador","goles","tiros","pases","centros"])
        self.start(1)
    def read(self,name,cols):
        p=self.folder/name
        return pd.read_csv(p) if p.exists() else pd.DataFrame(columns=cols)
    def start(self,mode,selecciones=("Local", "Visitante")):
        self.id=uuid.uuid4().hex[:12]; self.mode=mode; self.when=datetime.now()
        self.selecciones=selecciones
        self.score={0:0,1:0}; self.stats={}; self.done=False
    def event(self,team,player,action,seconds):
        row={"id":self.id,"segundo":round(seconds,2),"equipo":team,
             "pais":self.selecciones[team],"jugador":player,"evento":action}
        self.events=pd.concat([self.events,pd.DataFrame([row])],ignore_index=True)
    def action(self,team,player,action,seconds):
        if action not in ("tiro","pase","centro"): return
        key=(team,player)
        self.stats.setdefault(key,{"goles":0,"tiros":0,"pases":0,"centros":0})
        columna={"tiro":"tiros","pase":"pases","centro":"centros"}[action]
        self.stats[key][columna]+=1
        self.event(team,player,action,seconds)
    def goal(self,team,player,seconds):
        self.score[team]+=1; key=(team,player)
        self.stats.setdefault(key,{"goles":0,"tiros":0,"pases":0,"centros":0}); self.stats[key]["goles"]+=1
        self.event(team,player,"gol",seconds)
    def finish(self):
        if self.done: return
        result="empate" if self.score[0]==self.score[1] else ("gana_equipo_0" if self.score[0]>self.score[1] else "gana_equipo_1")
        row={"id":self.id,"fecha":self.when.isoformat(timespec="seconds"),"modo":MODOS[self.mode],
             "pais_0":self.selecciones[0],"pais_1":self.selecciones[1],
             "goles_0":self.score[0],"goles_1":self.score[1],"resultado":result}
        self.matches=pd.concat([self.matches,pd.DataFrame([row])],ignore_index=True)
        rows=[{"id":self.id,"equipo":t,"pais":self.selecciones[t],"jugador":p,**s}
              for (t,p),s in self.stats.items()]
        if rows: self.players=pd.concat([self.players,pd.DataFrame(rows)],ignore_index=True)
        self.matches.to_csv(self.folder/"partidos.csv",index=False); self.events.to_csv(self.folder/"eventos.csv",index=False); self.players.to_csv(self.folder/"jugadores.csv",index=False)
        self.done=True


from dataclasses import dataclass


@dataclass(frozen=True)
class Seleccion:
    nombre: str
    sigla: str
    camiseta: tuple[int, int, int]
    detalle: tuple[int, int, int]
    bandera: tuple[tuple[int, int, int], ...]
    once: tuple[str, ...]


# Nombres ficticios para que cada selección muestre una plantilla propia sin
# depender de licencias de jugadores, clubes, escudos ni marcas comerciales.
SELECCIONES = (
    Seleccion("Argentina", "ARG", (91, 177, 220), (245, 245, 237), ((116, 192, 231), (248, 248, 239), (116, 192, 231)),
              ("Mateo Ríos", "Tomás Ibarra", "Julián Acosta", "Nicolás Vera", "Bruno Sosa", "Thiago Luna", "Franco Paredes", "Lautaro Gil", "Damián Cruz", "Santiago Moya", "Benjamín Solís")),
    Seleccion("Brasil", "BRA", (245, 202, 43), (35, 143, 82), ((36, 147, 78), (245, 205, 49), (39, 79, 157)),
              ("Caio Nunes", "Rafael Costa", "Davi Almeida", "Enzo Rocha", "João Viana", "Lucas Melo", "Pedro Reis", "Iago Freitas", "Natan Lima", "Vitor Alves", "Gabriel Moraes")),
    Seleccion("Colombia", "COL", (226, 48, 52), (246, 205, 49), ((242, 205, 47), (35, 76, 157), (204, 48, 49)),
              ("Samuel Rojas", "David Rincón", "Juan Cárdenas", "Andrés Pardo", "Kevin Salazar", "Mateo Beltrán", "Felipe Torres", "Nicolás Ospina", "Sergio Lozano", "Daniel Buitrago", "Esteban Murillo")),
    Seleccion("España", "ESP", (203, 48, 49), (242, 198, 55), ((196, 43, 48), (238, 197, 53), (196, 43, 48)),
              ("Álvaro León", "Diego Campos", "Pablo Rivas", "Mario Soler", "Hugo Marín", "Adrián Vidal", "Iker Santos", "Marco Gil", "Sergio Alba", "Raúl Vega", "Leo Navarro")),
    Seleccion("Francia", "FRA", (48, 78, 160), (239, 241, 239), ((46, 69, 145), (239, 239, 232), (202, 54, 58)),
              ("Noah Laurent", "Lucas Petit", "Ethan Moreau", "Hugo Bernard", "Enzo Robert", "Adam Richard", "Léo Dubois", "Mathis Simon", "Nolan Michel", "Tom Leroy", "Yanis Roux")),
    Seleccion("Alemania", "GER", (232, 232, 222), (53, 58, 65), ((32, 32, 34), (219, 55, 51), (231, 195, 52)),
              ("Felix Weber", "Lukas Fischer", "Jonas Becker", "Paul Hoffmann", "Ben Schneider", "Leon Wagner", "Noah Meyer", "Elias Schulz", "Finn Koch", "Max Braun", "Theo Richter")),
    Seleccion("Inglaterra", "ENG", (238, 239, 233), (50, 75, 135), ((238, 238, 232), (203, 50, 53), (238, 238, 232)),
              ("Oliver Smith", "Harry Jones", "Jack Taylor", "Charlie Brown", "George Wilson", "Leo Davies", "Alfie Evans", "Theo Thomas", "Freddie Moore", "Oscar Hall", "Arthur Clarke")),
    Seleccion("México", "MEX", (43, 139, 90), (239, 239, 227), ((36, 123, 75), (238, 238, 229), (201, 54, 53)),
              ("Santiago Flores", "Emiliano Cruz", "Diego Herrera", "Sebastián Reyes", "Gael Navarro", "Leonardo Vega", "Matías Aguilar", "Iker Castillo", "Alexis Mendoza", "Bruno Campos", "Joaquín Fuentes")),
    Seleccion("Uruguay", "URU", (88, 169, 215), (241, 241, 230), ((242, 242, 234), (92, 167, 207), (242, 242, 234)),
              ("Martín Silva", "Facundo Pérez", "Nicolás Castro", "Thiago Ramos", "Agustín Suárez", "Franco Méndez", "Luca Pereira", "Mateo Cabrera", "Bruno Núñez", "Valentín Díaz", "Santino Olivera")),
    Seleccion("Japón", "JPN", (239, 239, 231), (53, 79, 143), ((240, 240, 235), (204, 51, 58), (240, 240, 235)),
              ("Haruto Sato", "Ren Suzuki", "Yuto Takahashi", "Sota Tanaka", "Kaito Watanabe", "Riku Ito", "Minato Nakamura", "Aoi Kobayashi", "Yuma Kato", "Itsuki Yoshida", "Hinata Yamada")),
    Seleccion("Portugal", "POR", (41, 128, 77), (208, 53, 52), ((30, 117, 69), (207, 49, 51), (207, 49, 51)),
              ("Tiago Silva", "Gonçalo Costa", "Rafael Santos", "Diogo Pereira", "Afonso Ferreira", "João Oliveira", "Martim Rodrigues", "Francisco Martins", "Duarte Sousa", "Tomás Lopes", "Miguel Gomes")),
)

# (avance desde el arco propio, carril lateral); índice 0 es arquero.
FORMACION_433 = (
    (.055, 0.0),
    (.205, -.73), (.225, -.25), (.225, .25), (.205, .73),
    (.405, -.52), (.435, 0.0), (.405, .52),
    (.685, -.68), (.735, 0.0), (.685, .68),
)


def obtener_seleccion(indice: int) -> Seleccion:
    return SELECCIONES[indice % len(SELECCIONES)]


def formacion(equipo: int, indice: int, x_min: float, x_max: float,
              y_centro: float, y_radio: float) -> tuple[float, float]:
    """Convierte la formación 4-3-3 a coordenadas; ambos atacan hacia el centro rival."""
    avance, carril = FORMACION_433[indice % 11]
    x = x_min + avance * (x_max - x_min) if equipo == 0 else x_max - avance * (x_max - x_min)
    return x, y_centro + carril * y_radio


def color_equipo(indice_pais: int, equipo: int = 0) -> tuple[int, int, int]:
    seleccion = obtener_seleccion(indice_pais)
    return seleccion.camiseta if equipo == 0 else seleccion.detalle


import atexit
import math
import os
import queue
import random
import struct
import subprocess
import sys
import threading
import wave
from pathlib import Path


CARPETA_AUDIO = Path(__file__).resolve().parent / "assets" / "audio"


class AudioFutbol:
    """Reproduce sonidos con Pygame y delega TTS para no bloquear el juego."""

    def __init__(self) -> None:
        self.sonidos: dict[str, pygame.mixer.Sound] = {}
        self.ultimo_poseedor: str | None = None
        self.ultimo_anuncio = -10.0
        self.cola_voz: queue.Queue[str | None] = queue.Queue(maxsize=6)
        self.reproductor_voz: subprocess.Popen | None = None
        self.efectos_activos = False
        self.cerrado = False
        self._iniciar_efectos()
        # La voz se inicia aunque falle el dispositivo de mezcla de efectos.
        self._iniciar_voz()
        atexit.register(self.cerrar)

    @staticmethod
    def _crear_wav(ruta: Path, tipo: str, duracion: float,
                   frecuencia: int = 22050) -> None:
        """Crea recursos de audio del juego sin conexión ni archivos licenciados."""
        ruta.parent.mkdir(parents=True, exist_ok=True)
        rng = random.Random({"estadio": 31, "gol": 73, "silbato": 19, "balon": 47}[tipo])
        datos = bytearray()
        ruido_filtrado = 0.0
        total = max(1, round(duracion * frecuencia))
        for muestra_indice in range(total):
            t = muestra_indice / frecuencia
            borde = min(1.0, t / .28, (duracion - t) / .28)
            borde = max(0.0, borde)
            if tipo == "estadio":
                # Ruido rosa con tres bandas de público y oleadas de intensidad.
                ruido_filtrado = ruido_filtrado * .955 + rng.uniform(-1, 1) * .045
                oleada = .52 + .20 * math.sin(t * .77) + .11 * math.sin(t * 1.93)
                murmullo = .12 * math.sin(math.tau * 185 * t + .3 * math.sin(t * 2.1))
                cantos = .075 * math.sin(math.tau * 310 * t + .4 * math.sin(t * 1.3))
                valor = (ruido_filtrado * .74 + murmullo + cantos) * oleada * borde
            elif tipo == "gol":
                subida = min(1.0, t * 2.8)
                caida = max(0.0, 1 - t / duracion * .65)
                envolvente = subida * caida
                ruido_filtrado = ruido_filtrado * .90 + rng.uniform(-1, 1) * .10
                grito = (.22 * math.sin(math.tau * (270 + 26 * math.sin(t * 3)) * t)
                         + .14 * math.sin(math.tau * (410 + 17 * math.sin(t * 2)) * t))
                valor = (ruido_filtrado * .38 + grito) * envolvente * borde
            elif tipo == "silbato":
                envolvente = min(1.0, t * 28) * max(0.0, min(1.0, (duracion - t) * 13))
                valor = .64 * math.sin(math.tau * (1880 + 75 * math.sin(t * 8)) * t) * envolvente
            else:
                golpe = math.exp(-t * 33)
                valor = (rng.uniform(-1, 1) * .56 + .35 * math.sin(math.tau * 125 * t)) * golpe
            pcm = int(max(-.95, min(.95, valor)) * 32767)
            datos.extend(struct.pack("<h", pcm))
        with wave.open(str(ruta), "wb") as archivo:
            archivo.setnchannels(1)
            archivo.setsampwidth(2)
            archivo.setframerate(frecuencia)
            archivo.writeframes(datos)

    def _iniciar_efectos(self) -> None:
        try:
            import pygame
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=1024)
            pygame.mixer.set_num_channels(16)
            duraciones = {"estadio": 7.0, "gol": 3.4, "silbato": .75, "balon": .28}
            for nombre, duracion in duraciones.items():
                archivo = CARPETA_AUDIO / f"{nombre}.wav"
                if not archivo.exists():
                    self._crear_wav(archivo, nombre, duracion)
                self.sonidos[nombre] = pygame.mixer.Sound(str(archivo))
            # El sonido de tribuna se oye claramente, sin tapar los comentarios.
            self.sonidos["estadio"].set_volume(.46)
            self.sonidos["gol"].set_volume(.92)
            self.sonidos["silbato"].set_volume(.62)
            self.sonidos["balon"].set_volume(.76)
            self.sonidos["estadio"].play(loops=-1)
            self.efectos_activos = True
        except Exception as error:
            self._registrar_estado_voz(f"Mezclador de audio no disponible: {error}")
            # Plan B en Windows: al menos suena la tribuna mediante winsound.
            if os.name == "nt":
                try:
                    import winsound
                    winsound.PlaySound(str(CARPETA_AUDIO / "estadio.wav"),
                                       winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_LOOP)
                    self.efectos_activos = True
                except Exception:
                    self.efectos_activos = False

    @staticmethod
    def _registrar_estado_voz(estado: str) -> None:
        ruta = CARPETA_AUDIO.parent / "narrador_estado.txt"
        try:
            ruta.parent.mkdir(parents=True, exist_ok=True)
            ruta.write_text(estado, encoding="utf-8")
        except OSError:
            pass

    def _iniciar_voz(self) -> None:
        """Inicia el comentarista en un proceso Python en línea (sin otro .py)."""
        codigo = r"""
import os, sys, subprocess
from pathlib import Path

def guardar_estado(texto):
    try:
        ruta = Path.cwd() / "assets" / "narrador_estado.txt"
        ruta.parent.mkdir(parents=True, exist_ok=True)
        ruta.write_text(texto, encoding="utf-8")
    except OSError:
        pass

def voz_sistema_windows():
    comando = (
        "Add-Type -AssemblyName System.Speech; "
        "[Console]::InputEncoding=New-Object System.Text.UTF8Encoding($false); "
        "$v=New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        "try{$v.SelectVoiceByHints([System.Speech.Synthesis.VoiceGender]::NotSet, "
        "[System.Speech.Synthesis.VoiceAge]::NotSet, 0, "
        "[System.Globalization.CultureInfo]::GetCultureInfo('es-CO'))}catch{}; "
        "while(($s=[Console]::In.ReadLine()) -ne $null){if($s){$v.Speak($s)}}; $v.Dispose()"
    )
    powershell = os.environ.get("WINDIR", r"C:\Windows") + r"\System32\WindowsPowerShell\v1.0\powershell.exe"
    proceso = subprocess.Popen([powershell, "-NoProfile", "-Command", comando],
                               stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, text=True, encoding="utf-8",
                               bufsize=1)
    guardar_estado("Narrador activo: System.Speech de Windows")
    for frase in sys.stdin:
        if proceso.poll() is not None:
            break
        if proceso.stdin:
            proceso.stdin.write(frase)
            proceso.stdin.flush()
    if proceso.stdin:
        proceso.stdin.close()
    return proceso.wait()

def main():
    try:
        import pyttsx3
        motor = pyttsx3.init("sapi5" if os.name == "nt" else None)
        motor.setProperty("rate", 175)
        motor.setProperty("volume", 1.0)
        for voz in motor.getProperty("voices") or []:
            info = f"{voz.id} {voz.name}".lower()
            if any(p in info for p in ("spanish", "español", "es-es", "es-mx", "es_")):
                motor.setProperty("voice", voz.id)
                break
        guardar_estado("Narrador activo: pyttsx3/SAPI")
        for frase in sys.stdin:
            frase = frase.strip()
            if frase:
                motor.say(frase)
                motor.runAndWait()
        motor.stop()
    except Exception as error:
        guardar_estado(f"pyttsx3 no inició; usando voz del sistema: {error}")
        if os.name == "nt":
            try:
                voz_sistema_windows()
                return
            except Exception as error_windows:
                guardar_estado(f"Narrador no disponible: {error_windows}")
        else:
            guardar_estado(f"Narrador no disponible: {error}")

if __name__ == "__main__":
    main()
"""
        try:
            self.reproductor_voz = subprocess.Popen(
                [sys.executable, "-u", "-c", codigo], stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                text=True, encoding="utf-8", bufsize=1,
                cwd=str(Path(__file__).resolve().parent),
            )
            hilo = threading.Thread(target=self._enviar_textos_voz, daemon=True,
                                    name="comentarista-futbol")
            hilo.start()
        except Exception as error:
            self._registrar_estado_voz(f"No se pudo iniciar la voz: {error}")

    def _enviar_textos_voz(self) -> None:
        proceso = self.reproductor_voz
        if proceso is None or proceso.stdin is None:
            return
        try:
            while True:
                texto = self.cola_voz.get()
                if texto is None or proceso.poll() is not None:
                    break
                proceso.stdin.write(texto.replace("\n", " ") + "\n")
                proceso.stdin.flush()
        except (BrokenPipeError, OSError, ValueError):
            pass
        finally:
            try:
                proceso.stdin.close()
            except (OSError, ValueError):
                pass

    def narrar(self, texto: str) -> None:
        try:
            self.cola_voz.put_nowait(texto)
        except queue.Full:
            try:
                self.cola_voz.get_nowait()
                self.cola_voz.put_nowait(texto)
            except queue.Empty:
                pass

    def posesion(self, nombre: str, pais: str, ahora: float) -> None:
        clave = f"{pais}:{nombre}"
        if clave != self.ultimo_poseedor and ahora - self.ultimo_anuncio >= 1.0:
            self.ultimo_poseedor = clave
            self.ultimo_anuncio = ahora
            self.narrar(f"{nombre}, de {pais}, conduce el balón.")

    def efecto(self, nombre: str) -> None:
        sonido = self.sonidos.get(nombre)
        if sonido is not None:
            try:
                sonido.play()
            except Exception:
                pass
        elif os.name == "nt" and nombre in ("gol", "silbato", "balon"):
            try:
                import winsound
                archivo = CARPETA_AUDIO / f"{nombre}.wav"
                winsound.PlaySound(str(archivo), winsound.SND_FILENAME | winsound.SND_ASYNC)
            except Exception:
                pass

    def gol(self, nombre: str, pais: str) -> None:
        self.efecto("gol")
        self.ultimo_poseedor = None
        self.narrar(f"¡Gooooool! ¡Gol de {nombre}, para {pais}!")

    def jugada(self, nombre: str, accion: str) -> None:
        # Si el mismo jugador recupera el balón después de una acción,
        # debe volver a anunciarse su posesión.
        self.ultimo_poseedor = None
        frases = {
            "tiro": f"¡Remate de {nombre}!",
            "pase": f"Pase de {nombre}.",
            "centro": f"¡Centro al área de {nombre}!",
        }
        if accion in frases:
            self.narrar(frases[accion])

    def cerrar(self) -> None:
        if self.cerrado:
            return
        self.cerrado = True
        sonido_estadio = self.sonidos.get("estadio")
        if sonido_estadio is not None:
            try:
                sonido_estadio.stop()
            except Exception:
                pass
        if os.name == "nt":
            try:
                import winsound
                winsound.PlaySound(None, 0)
            except Exception:
                pass
        try:
            while True:
                self.cola_voz.get_nowait()
        except queue.Empty:
            self.cola_voz.put_nowait(None)
        if self.reproductor_voz is not None:
            try:
                self.reproductor_voz.wait(timeout=.4)
            except subprocess.TimeoutExpired:
                self.reproductor_voz.terminate()


import colorsys
import math
from pathlib import Path

import pygame


# Cambia a "3d" para usar los sprites prerenderizados de assets_3d.
ASSET_STYLE = "2d"
if ASSET_STYLE not in {"2d", "3d"}:
    raise ValueError('ASSET_STYLE debe ser "2d" o "3d".')


class RecursosVisuales:
    """Carga PNG, recorta sprites aislados y ofrece alternativas seguras."""

    ALTO_JUGADOR = 42
    TAMANO_BALON = 22

    def __init__(self) -> None:
        self.raiz = Path(__file__).resolve().parent / f"assets_{ASSET_STYLE}"
        self.errores: list[str] = []
        self.cancha: pygame.Surface | None = None
        self.balon: pygame.Surface | None = None
        self.hinchas: pygame.Surface | None = None
        self.jugadores: dict[int, list[pygame.Surface]] = {0: [], 1: []}
        # Un recorte fijo por equipo: verde para el primero y azul para el segundo.
        self.porteros: dict[int, pygame.Surface | None] = {0: None, 1: None}
        self._cache_portero: dict[tuple[int, int], pygame.Surface] = {}
        self._cache_previa: dict[tuple[int, int], pygame.Surface] = {}
        self._cache_jugador_base: dict[
            tuple[int, int, tuple[int, int, int]], pygame.Surface
        ] = {}
        self._cache_jugador: dict[
            tuple[int, int, int, tuple[int, int, int]], pygame.Surface
        ] = {}
        self._cache_balon: pygame.Surface | None = None
        self._cache_hinchas: dict[tuple[int, int], pygame.Surface] = {}

        # La cancha cenital ilustrada encaja con el estilo 2D. La versión 3D
        # del recurso es trapezoidal y se deja a la cancha vectorial actual.
        if ASSET_STYLE == "2d":
            self.cancha = self._cargar_png("cancha/cancha_completa.png")

        hojas = {
            0: "jugadores/jugadores_amarillo_azul.png",
            1: "jugadores/jugadores_rojo_blanco.png",
        }
        for equipo, nombre in hojas.items():
            hoja = self._cargar_png(nombre)
            if hoja is not None:
                self.jugadores[equipo] = self._recortar_figuras(hoja, nombre)

        hoja_porteros = self._cargar_png("jugadores/porteros_y_arbitros.png")
        if hoja_porteros is not None:
            figuras_portero = self._recortar_figuras(hoja_porteros, "porteros_y_arbitros.png")
            # Las dos primeras siluetas de la hoja son porteros con uniformes distintos.
            for equipo, figura in enumerate(figuras_portero[:2]):
                self.porteros[equipo] = figura
            if len(figuras_portero) < 2:
                self.errores.append("Porteros: la hoja no contiene dos recortes utilizables")

        self.balon = self._cargar_png("balon_y_efectos/balones.png")
        self.hinchas = self._cargar_png("hinchas/hinchas_fila_1.png")

    def _cargar_png(self, relativo: str) -> pygame.Surface | None:
        """Carga PNG con alfa; ante un fallo, conserva la vista vectorial."""
        ruta = self.raiz / relativo
        try:
            return pygame.image.load(ruta).convert_alpha()
        except (pygame.error, FileNotFoundError, OSError) as error:
            self.errores.append(f"{ruta.name}: {error}")
            return None

    def _recortar_figuras(self, hoja: pygame.Surface,
                          nombre: str) -> list[pygame.Surface]:
        """Detecta siluetas por componentes alfa, sin inventar coordenadas."""
        try:
            mascara = pygame.mask.from_surface(hoja, 96)
            componentes = mascara.connected_components(minimum=80)
            limites: list[pygame.Rect] = []
            for componente in componentes:
                rectangulos = cast(list[pygame.Rect], componente.get_bounding_rects())
                if not rectangulos:
                    continue
                caja = rectangulos[0]
                # Excluye balones, confeti y otros detalles desconectados.
                if caja.height >= 42 and caja.width >= 26 and caja.y < hoja.get_height() * .8:
                    limites.append(caja)

            figuras: list[pygame.Surface] = []
            for caja in sorted(limites, key=lambda r: (r.x, r.y)):
                margen = caja.inflate(4, 4).clip(hoja.get_rect())
                figuras.append(hoja.subsurface(margen).copy())
            if not figuras:
                self.errores.append(f"{nombre}: no se detectaron sprites independientes")
            return figuras
        except (pygame.error, ValueError) as error:
            self.errores.append(f"{nombre}: no se pudieron recortar sprites ({error})")
            return []

    def tiene_jugadores(self, equipo: int) -> bool:
        return bool(self.jugadores.get(equipo))

    def dibujar_vista_previa_cancha(self, destino: pygame.Surface,
                                    rect: pygame.Rect) -> bool:
        """Muestra la imagen de cancha en el menú sin deformar su proporción."""
        if self.cancha is None:
            return False
        try:
            factor = min(rect.width / self.cancha.get_width(),
                         rect.height / self.cancha.get_height())
            tamano = (max(1, round(self.cancha.get_width() * factor)),
                      max(1, round(self.cancha.get_height() * factor)))
            if tamano not in self._cache_previa:
                self._cache_previa[tamano] = pygame.transform.smoothscale(self.cancha, tamano)
            imagen = self._cache_previa[tamano]
            destino.blit(imagen, imagen.get_rect(center=rect.center))
            return True
        except pygame.error as error:
            self.errores.append(f"Cancha: no se pudo escalar la vista previa ({error})")
            self.cancha = None
            return False

    def dibujar_jugador(self, destino: pygame.Surface, equipo: int,
                        indice: int, xy: tuple[int, int],
                        angulo: float,
                        color_uniforme: tuple[int, int, int] | None = None) -> bool:
        """Dibuja sprites completos, erguidos y orientados sin girar el cuerpo."""
        x, y = xy
        orientacion = int(math.cos(angulo) < 0)

        # El portero usa su uniforme propio; nunca se tiñe con la camiseta de campo.
        if indice == 0 and self.porteros.get(equipo) is not None:
            clave_portero = (equipo, orientacion)
            if clave_portero not in self._cache_portero:
                original = self.porteros[equipo]
                assert original is not None
                alto = self.ALTO_JUGADOR
                ancho = max(20, round(original.get_width() * alto / original.get_height()))
                base = pygame.transform.smoothscale(original, (ancho, alto))
                if orientacion:
                    base = pygame.transform.flip(base, True, False)
                self._cache_portero[clave_portero] = base
            sprite = self._cache_portero[clave_portero]
            destino.blit(sprite, sprite.get_rect(center=xy))
            return True

        figuras = self.jugadores.get(equipo, [])
        if not figuras:
            return False
        numero = indice % len(figuras)
        color = color_uniforme or (255, 255, 255)
        clave_base = (equipo, numero, color)
        clave = (equipo, numero, orientacion, color)
        if clave_base not in self._cache_jugador_base:
            original = figuras[numero]
            alto = self.ALTO_JUGADOR
            ancho = max(20, round(original.get_width() * alto / original.get_height()))
            base = pygame.transform.smoothscale(original, (ancho, alto))
            tono_objetivo, saturacion_objetivo, valor_objetivo = colorsys.rgb_to_hsv(
                *(componente / 255 for componente in color))
            for py in range(base.get_height()):
                for px in range(base.get_width()):
                    r, g, b, alpha = base.get_at((px, py))
                    tono, saturacion, valor = colorsys.rgb_to_hsv(
                        r / 255, g / 255, b / 255)
                    # Recolorea la camiseta fuente y conserva piel, shorts y detalles.
                    camiseta = (0.095 <= tono <= 0.19 if equipo == 0
                                else tono <= 0.04 or tono >= 0.96)
                    if alpha and saturacion > 0.35 and camiseta:
                        nuevo = colorsys.hsv_to_rgb(
                            tono_objetivo,
                            saturacion_objetivo * (0.45 + 0.55 * saturacion),
                            valor_objetivo * (0.45 + 0.55 * valor),
                        )
                        base.set_at((px, py), tuple(round(c * 255) for c in nuevo) + (alpha,))
            self._cache_jugador_base[clave_base] = base
        if clave not in self._cache_jugador:
            base = self._cache_jugador_base[clave_base]
            # Los sprites son ilustraciones cenitales 2D: reflejar evita tumbas
            # visuales y deformación que produce una rotación libre de 360 grados.
            self._cache_jugador[clave] = (
                pygame.transform.flip(base, True, False) if orientacion else base
            )
        sprite = self._cache_jugador[clave]
        destino.blit(sprite, sprite.get_rect(center=xy))
        return True

    def dibujar_balon(self, destino: pygame.Surface,
                      xy: tuple[int, int]) -> bool:
        """Dibuja el primer balón aislado de la tira como sprite individual."""
        if self.balon is None:
            return False
        if self._cache_balon is None:
            try:
                mascara = pygame.mask.from_surface(self.balon, 96)
                candidatos: list[pygame.Rect] = []
                for componente in mascara.connected_components(minimum=80):
                    rectangulos = cast(list[pygame.Rect], componente.get_bounding_rects())
                    if not rectangulos:
                        continue
                    caja = rectangulos[0]
                    proporcion = caja.width / max(1, caja.height)
                    if caja.width >= 30 and caja.height >= 30 and .7 <= proporcion <= 1.35:
                        candidatos.append(caja)
                if not candidatos:
                    self.errores.append("Balón: no se encontró un recorte circular; se usa el vectorial")
                    self.balon = None
                    return False
                caja = min(candidatos, key=lambda r: (r.x, r.y))
                sprite = self.balon.subsurface(caja.inflate(4, 4).clip(self.balon.get_rect())).copy()
                self._cache_balon = pygame.transform.smoothscale(
                    sprite, (self.TAMANO_BALON, self.TAMANO_BALON))
            except (pygame.error, ValueError) as error:
                self.errores.append(f"Balón: no se pudo recortar ({error})")
                self.balon = None
                return False
        destino.blit(self._cache_balon, self._cache_balon.get_rect(center=xy))
        return True

    def dibujar_hinchas(self, destino: pygame.Surface,
                        rect: pygame.Rect) -> bool:
        """Usa la fila completa como decoración, nunca como personas sueltas."""
        if self.hinchas is None:
            return False
        try:
            clave = rect.size
            if clave not in self._cache_hinchas:
                factor = min(rect.width / self.hinchas.get_width(),
                             rect.height / self.hinchas.get_height())
                tamano = (max(1, round(self.hinchas.get_width() * factor)),
                          max(1, round(self.hinchas.get_height() * factor)))
                self._cache_hinchas[clave] = pygame.transform.smoothscale(self.hinchas, tamano)
            imagen = self._cache_hinchas[clave]
            destino.blit(imagen, imagen.get_rect(center=rect.center))
            return True
        except pygame.error as error:
            self.errores.append(f"Hinchas: no se pudo escalar ({error})")
            self.hinchas = None
            return False

# ===== Juego 2D =====

ANCHO, ALTO = 1280, 720
FPS = 60
CAMPO = pygame.Rect(70, 130, 1140, 490)
VERDE_1 = (26, 111, 68)
VERDE_2 = (31, 124, 74)
BLANCO = (239, 246, 241)
AZUL = (33, 139, 255)
ROJO = (247, 74, 83)
AMARILLO = (255, 206, 76)
FONDO = (8, 17, 29)


def limitar_posicion(x: float, y: float) -> tuple[float, float]:
    return (clamp(x, CAMPO.left + 22, CAMPO.right - 22),
            clamp(y, CAMPO.top + 22, CAMPO.bottom - 22))


@dataclass
class Jugador:
    equipo: int
    indice: int
    control: str
    nombre: str
    x: float
    y: float
    casa: tuple[float, float]
    objetivo: tuple[float, float]
    angulo: float = 0.0
    energia: float = 1.0
    recarga_robo: float = 0.0


class Futbol2D:
    """Partido 2D con controles locales, IA táctica y registro CSV."""

    def __init__(self) -> None:
        pygame.init()
        pygame.display.set_caption("PITCH / Fútbol 2D con IA")
        self.pantalla = pygame.display.set_mode((ANCHO, ALTO))
        self.recursos_visual = RecursosVisuales()
        self.reloj = pygame.time.Clock()
        self.fuentes = {
            "mini": pygame.font.SysFont("segoeui", 14),
            "normal": pygame.font.SysFont("segoeui", 18),
            "bold": pygame.font.SysFont("segoeui", 22, bold=True),
            "titulo": pygame.font.SysFont("segoeui", 48, bold=True),
            "numero": pygame.font.SysFont("segoeui", 18, bold=True),
        }
        self.estado = "menu"
        self.modo = 1
        self.ia: IAFutbol | None = None
        self.jugadores: list[Jugador] = []
        self.balon = pygame.Vector2(CAMPO.center)
        self.vel_balon = pygame.Vector2()
        self.poseedor: Jugador | None = None
        self.altura = 0.0
        self.vel_altura = 0.0
        self.tiempo = 0.0
        self.reloj_partido = 150.0
        self.temporizador_ia = 0.0
        self.pausa = False
        self.fin = False
        self.mensaje = ""
        self.mensaje_tiempo = 0.0
        self.numero_jugadas = 0
        self.pais_indices = [0, 1]
        self.audio = AudioFutbol()
        self.ultimo_golpeador: Jugador | None = None
        self._poseedor_control_anterior: Jugador | None = None
        self.botones_overlay: dict[str, pygame.Rect] = {}
        self.ejecutando = True
        self._iniciar_fuentes()

    def _iniciar_fuentes(self) -> None:
        # Mantiene un juego legible al cambiar el tamaño de la ventana.
        self.fuentes["mini"] = pygame.font.SysFont("segoeui", 14)

    def iniciar_partido(self, modo: int) -> None:
        self.modo = modo
        if self.ia is None:
            self.ia = IAFutbol()
        elif not self.ia.log.done:
            self.ia.log.finish()
        paises = tuple(obtener_seleccion(i).nombre for i in self.pais_indices)
        self.ia.log.start(modo, paises)
        self.jugadores = []
        for equipo in (0, 1):
            seleccion = obtener_seleccion(self.pais_indices[equipo])
            for indice in range(11):
                control = "IA"
                if modo == 1 and equipo == 0 and indice == 10:
                    control = "H1"
                elif modo == 2 and indice == 10:
                    control = "H1" if equipo == 0 else "H2"
                elif modo in (3, 4) and equipo == 1 and indice in (9, 10):
                    control = "H2" if indice == 9 else "H1"
                x, y = formacion(equipo, indice, CAMPO.left + 30, CAMPO.right - 30,
                                 CAMPO.centery, CAMPO.height * .40)
                casa = (x, y)
                self.jugadores.append(Jugador(equipo, indice, control,
                                              seleccion.once[indice], x, y, casa, casa,
                                              angulo=0.0 if equipo == 0 else math.pi))
        self._iniciar_saque(0)
        self.tiempo = self.temporizador_ia = 0.0
        self.reloj_partido = 150.0
        self.pausa = self.fin = False
        self.estado = "partido"
        self._poseedor_control_anterior = None
        self.mensaje = "¡Que ruede el balón!"
        self.mensaje_tiempo = 2.2
        self.audio.efecto("silbato")

    def _iniciar_saque(self, equipo: int) -> None:
        """Reubica los equipos y entrega el saque al equipo que debe iniciar."""
        signo = 1 if equipo == 0 else -1
        for jugador in self.jugadores:
            jugador.x, jugador.y = jugador.casa
            jugador.objetivo = jugador.casa
            jugador.angulo = 0.0 if jugador.equipo == 0 else math.pi
            jugador.recarga_robo = 0.0

        sacador = max((p for p in self.jugadores if p.equipo == equipo
                       and p.indice >= 9), key=lambda p: p.indice, default=None)
        if sacador is not None:
            sacador.x = CAMPO.centerx - signo * 19
            sacador.y = CAMPO.centery
            sacador.objetivo = (sacador.x, sacador.y)
            sacador.angulo = 0.0 if equipo == 0 else math.pi
            sacador.recarga_robo = .8
        self.poseedor = sacador
        self.balon.update(CAMPO.center)
        self.vel_balon.update(0, 0)
        self.altura = self.vel_altura = 0.0
        self.ultimo_golpeador = None
        self._poseedor_control_anterior = None

    def volver_al_menu(self) -> None:
        if self.ia and not self.ia.log.done:
            self.ia.log.finish()
        self.estado = "menu"

    def bucle(self) -> None:
        try:
            while self.ejecutando:
                # tick limita la velocidad a 60 FPS y espera el siguiente frame.
                dt = min(self.reloj.tick(FPS) / 1000.0, .035)
                self._eventos()
                if self.estado == "partido" and not self.pausa and not self.fin:
                    self.actualizar(dt)
                self.dibujar()
                pygame.display.flip()
        except KeyboardInterrupt:
            # Ctrl+C o el botón Stop del IDE cierran el juego sin traceback.
            self.ejecutando = False
        finally:
            self.audio.cerrar()
            pygame.quit()

    def _eventos(self) -> None:
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                self.ejecutando = False
            elif evento.type == pygame.KEYDOWN:
                if evento.key == pygame.K_ESCAPE:
                    if self.estado == "partido":
                        if self.pausa or self.fin:
                            self.volver_al_menu()
                        else:
                            self.pausa = True
                    else:
                        self.ejecutando = False
                elif self.estado == "menu" and pygame.K_1 <= evento.key <= pygame.K_4:
                    self.iniciar_partido(evento.key - pygame.K_0)
                elif self.estado == "menu" and evento.key in (pygame.K_a, pygame.K_d,
                                                               pygame.K_LEFT, pygame.K_RIGHT):
                    equipo = 0 if evento.key in (pygame.K_a, pygame.K_d) else 1
                    paso = 1 if evento.key in (pygame.K_d, pygame.K_RIGHT) else -1
                    self.pais_indices[equipo] = (self.pais_indices[equipo] + paso) % len(SELECCIONES)
                elif self.estado == "menu" and evento.key == pygame.K_r:
                    equipo_ia = 1 if self.modo in (1, 2) else 0
                    opciones = [i for i in range(len(SELECCIONES))
                                if i != self.pais_indices[1 - equipo_ia]]
                    self.pais_indices[equipo_ia] = random.choice(opciones)
                elif self.estado == "partido":
                    if evento.key == pygame.K_p and not self.fin:
                        self.pausa = not self.pausa
                    elif evento.key == pygame.K_r:
                        self.iniciar_partido(self.modo)
                    elif evento.key in (pygame.K_SPACE, pygame.K_q, pygame.K_e,
                                        pygame.K_RCTRL, pygame.K_RETURN, pygame.K_u) and not self.pausa and not self.fin:
                        self._accion_humana(evento.key)
                    elif evento.key in (pygame.K_f, pygame.K_RALT) and not self.pausa and not self.fin:
                        self._robar("H1" if evento.key == pygame.K_f else "H2")
                    elif evento.key == pygame.K_TAB and not self.pausa and not self.fin:
                        self._cambiar_jugador("H1")
                    elif evento.key == pygame.K_BACKSPACE and not self.pausa and not self.fin:
                        self._cambiar_jugador("H2")
            elif (evento.type == pygame.MOUSEBUTTONDOWN and evento.button == 1
                  and self.estado == "partido" and (self.pausa or self.fin)):
                for accion, rect in self.botones_overlay.items():
                    if rect.collidepoint(evento.pos):
                        if accion == "continuar":
                            self.pausa = False
                        elif accion in ("reiniciar", "jugar_de_nuevo"):
                            self.iniciar_partido(self.modo)
                        elif accion == "menu":
                            self.volver_al_menu()
                        elif accion == "salir":
                            self.ejecutando = False
                        break

    def _jugador_humano(self, control: str) -> Jugador | None:
        return next((p for p in self.jugadores if p.control == control), None)

    def _cambiar_jugador(self, control: str, objetivo: Jugador | None = None) -> None:
        """Transfiere el control humano y deja a la IA manejando al jugador anterior."""
        activo = self._jugador_humano(control)
        if activo is None:
            return
        if objetivo is activo:
            return
        ocupados = {p.indice for p in self.jugadores
                    if p.equipo == activo.equipo and p is not activo
                    and p.control in ("H1", "H2")}
        candidatos = [p for p in self.jugadores if p.equipo == activo.equipo
                      and p is not activo and p.indice not in ocupados]
        if not candidatos:
            return
        coincide = next((p for p in candidatos if p is objetivo), None)
        if coincide is not None:
            nuevo = coincide
        else:
            por_indice = sorted(candidatos, key=lambda p: p.indice)
            siguiente = next((p for p in por_indice if p.indice > activo.indice), por_indice[0])
            nuevo = siguiente
        activo.control = "IA"
        nuevo.control = control
        self.mensaje = f"CONTROL · {nuevo.nombre} · {nuevo.indice + 1}"
        self.mensaje_tiempo = 1.0

    def _sincronizar_controles(self) -> None:
        """Cambia al receptor del pase o al defensor más cercano al recuperar/perder el balón."""
        if self.poseedor is self._poseedor_control_anterior:
            return
        self._poseedor_control_anterior = self.poseedor
        controles = [c for c in ("H1", "H2") if self._jugador_humano(c)]
        controles.sort()
        control_del_portador: str | None = None
        poseedor = self.poseedor
        if poseedor is not None:
            portadores_humanos: list[tuple[str, Jugador]] = []
            for control in controles:
                jugador = self._jugador_humano(control)
                if jugador is not None and jugador.equipo == poseedor.equipo:
                    portadores_humanos.append((control, jugador))
            if portadores_humanos:
                # En cooperativo, solo una persona cambia al jugador que recibió el balón.
                # La otra conserva su jugador y puede desmarcarse o apoyar la jugada.
                control_del_portador, _ = min(
                    portadores_humanos,
                    key=lambda par: math.dist((par[1].x, par[1].y),
                                              (poseedor.x, poseedor.y)))
                self._cambiar_jugador(control_del_portador, poseedor)
        for control in controles:
            activo = self._jugador_humano(control)
            if activo is None or control == control_del_portador:
                continue
            if self.poseedor is not None and self.poseedor.equipo == activo.equipo:
                # El segundo usuario mantiene su jugador controlable independiente.
                continue
            if activo.indice == 0:
                # Si el usuario tomó al arquero, lo conserva mientras defiende.
                continue
            ya_controlados = {p.indice for p in self.jugadores
                              if p.equipo == activo.equipo and p.control in ("H1", "H2")
                              and p is not activo}
            opciones = [p for p in self.jugadores if p.equipo == activo.equipo
                        and p.indice != 0 and p is not activo
                        and p.indice not in ya_controlados]
            mas_cercano = min(opciones, key=lambda p: math.dist((p.x, p.y), self.balon),
                              default=None)
            if mas_cercano:
                self._cambiar_jugador(control, mas_cercano)

    def _accion_humana(self, tecla: int) -> None:
        control = "H1" if tecla in (pygame.K_SPACE, pygame.K_q, pygame.K_e) else "H2"
        p = self._jugador_humano(control)
        if not p:
            return
        accion = ("pase" if tecla in (pygame.K_q, pygame.K_RETURN) else
                  "centro" if tecla in (pygame.K_e, pygame.K_u) else "tiro")
        compañero = self._mejor_companero(p, solo_ataque=accion == "centro")
        self._patear(p, accion, compañero)

    def _mejor_companero(self, jugador: Jugador, solo_ataque: bool = False) -> Jugador | None:
        """Prefiere una línea de pase hacia delante y alejada de los defensores."""
        signo = 1 if jugador.equipo == 0 else -1
        candidatos = [q for q in self.jugadores if q.equipo == jugador.equipo and q is not jugador
                      and q.indice != 0]
        if solo_ataque:
            candidatos = [q for q in candidatos if signo * (q.x - jugador.x) > 0]
        rivales = [q for q in self.jugadores if q.equipo != jugador.equipo]
        def puntuacion(q: Jugador) -> float:
            avance = signo * (q.x - jugador.x)
            distancia = math.dist((q.x, q.y), (jugador.x, jugador.y))
            espacio = min((math.dist((q.x, q.y), (r.x, r.y)) for r in rivales), default=180)
            return avance * .52 - distancia * .14 + min(150, espacio) * .28
        return max(candidatos, key=puntuacion, default=None)

    def _robar(self, control: str) -> None:
        """Permite presionar al rival o reclamar un balón dividido."""
        p = self._jugador_humano(control)
        if p is None:
            return
        objetivo = self.poseedor
        distancia = math.dist((p.x, p.y),
                              (objetivo.x, objetivo.y) if objetivo else self.balon)
        if objetivo is not None and objetivo.equipo != p.equipo and distancia < 48:
            self.poseedor = p
            p.recarga_robo = .65
            self.vel_balon.update(0, 0)
            self.mensaje, self.mensaje_tiempo = "¡RECUPERACIÓN!", 1.0
        elif objetivo is None and distancia < 44:
            self.poseedor = p
            self.vel_balon.update(0, 0)

    def actualizar(self, dt: float) -> None:
        assert self.ia is not None
        self.tiempo += dt
        self.reloj_partido = max(0, self.reloj_partido - dt)
        self.mensaje_tiempo = max(0, self.mensaje_tiempo - dt)
        teclas = pygame.key.get_pressed()
        for jugador in self.jugadores:
            if jugador.control == "IA":
                continue
            if jugador.control == "H1":
                dx = float(teclas[pygame.K_d]) - float(teclas[pygame.K_a])
                dy = float(teclas[pygame.K_s]) - float(teclas[pygame.K_w])
                sprint = teclas[pygame.K_LSHIFT]
            else:
                dx = float(teclas[pygame.K_RIGHT]) - float(teclas[pygame.K_LEFT])
                dy = float(teclas[pygame.K_DOWN]) - float(teclas[pygame.K_UP])
                sprint = teclas[pygame.K_RSHIFT]
            magnitud = math.hypot(dx, dy)
            if magnitud:
                velocidad = 300 if sprint and jugador.energia > .08 else 228
                if self.poseedor is jugador:
                    velocidad *= .90  # El control del balón reduce un poco la velocidad.
                dx, dy = dx / magnitud, dy / magnitud
                jugador.x, jugador.y = limitar_posicion(jugador.x + dx * velocidad * dt,
                                                          jugador.y + dy * velocidad * dt)
                jugador.angulo = math.atan2(dy, dx)
                jugador.energia = max(0, jugador.energia - dt * (.24 if sprint else .08))
            else:
                jugador.energia = min(1, jugador.energia + dt * .16)

        for jugador in self.jugadores:
            jugador.recarga_robo = max(0.0, jugador.recarga_robo - dt)
        self._actualizar_ia(dt)
        self._actualizar_posesion(dt)
        self._sincronizar_controles()
        if self.poseedor is not None:
            pais = obtener_seleccion(self.pais_indices[self.poseedor.equipo]).nombre
            self.audio.posesion(self.poseedor.nombre, pais, self.tiempo)
        self._actualizar_balon(dt)
        if self.reloj_partido <= 0:
            self.fin = True
            self.ia.log.finish()
            self.mensaje = "Final del partido · R para revancha"

    def _actualizar_ia(self, dt: float) -> None:
        assert self.ia is not None
        self.temporizador_ia -= dt
        if self.temporizador_ia <= 0:
            self.temporizador_ia = .22
            por_equipo = {t: [p for p in self.jugadores if p.equipo == t and p.control == "IA"]
                          for t in (0, 1)}
            for equipo, miembros in por_equipo.items():
                if not miembros:
                    continue
                signo = 1 if equipo == 0 else -1
                aliados = [p for p in self.jugadores if p.equipo == equipo]
                portador = self.poseedor
                rivales = [p for p in self.jugadores if p.equipo != equipo]
                cazador = min(miembros, key=lambda p: math.dist((p.x, p.y), self.balon))
                pelota = ((self.balon.x - CAMPO.centerx) / (CAMPO.width / 2),
                          (self.balon.y - CAMPO.centery) / (CAMPO.height / 2), self.altura / 90)
                actor = ((cazador.x - CAMPO.centerx) / (CAMPO.width / 2),
                         (cazador.y - CAMPO.centery) / (CAMPO.height / 2))
                mate_p = self._mejor_companero(cazador)
                mate = ((mate_p.x - CAMPO.centerx) / (CAMPO.width / 2),
                        (mate_p.y - CAMPO.centery) / (CAMPO.height / 2)) if mate_p else None
                accion, prediccion = self.ia.decide(pelota, actor, mate, signo)

                for p in miembros:
                    es_portador = portador is p
                    if p.indice == 0:
                        x_meta = CAMPO.left + 28 if equipo == 0 else CAMPO.right - 28
                        y_meta = CAMPO.centery + (self.balon.y - CAMPO.centery) * .22
                        p.objetivo = limitar_posicion(x_meta, y_meta)
                    elif es_portador:
                        meta_x = CAMPO.right - 34 if equipo == 0 else CAMPO.left + 34
                        presion = min((math.dist((p.x, p.y), (r.x, r.y)) for r in rivales), default=999)
                        p.angulo = math.atan2(CAMPO.centery - p.y, signo)
                        distancia_balon = math.dist((p.x, p.y), self.balon)
                        companion = self._mejor_companero(p)
                        distancia_meta = abs(meta_x - p.x)
                        if distancia_balon < 54 and distancia_meta < 255:
                            self._patear(p, "tiro", companion)
                        elif (distancia_balon < 54 and companion and
                              ((presion < 58 and accion == "pase") or
                               (distancia_meta < 390 and abs(p.y - CAMPO.centery) > 120
                                and signo * (p.x - CAMPO.centerx) > 120))):
                            tipo = ("centro" if abs(p.y - CAMPO.centery) > 120 and
                                    signo * (p.x - CAMPO.centerx) > 120 else "pase")
                            self._patear(p, tipo, self._mejor_companero(
                                p, solo_ataque=tipo == "centro"))
                        else:
                            p.objetivo = limitar_posicion(p.x + signo * 105,
                                                          p.y + (CAMPO.centery - p.y) * .12)
                    elif portador is not None and portador.equipo != equipo:
                        portador_actual = portador
                        if p is min(miembros, key=lambda q: math.dist((q.x, q.y),
                                                                     (portador_actual.x, portador_actual.y))):
                            p.objetivo = (portador_actual.x, portador_actual.y)
                        else:
                            apoyo_x = p.casa[0] + (self.balon.x - CAMPO.centerx) * .12
                            p.objetivo = limitar_posicion(apoyo_x,
                                                          p.casa[1] + (self.balon.y - p.casa[1]) * .18)
                    elif portador is None and p is cazador:
                        p.objetivo = (self.balon.x, self.balon.y)
                        if accion == "perseguir":
                            p.objetivo = limitar_posicion(CAMPO.centerx + prediccion[0] * CAMPO.width / 2,
                                                          CAMPO.centery + prediccion[1] * CAMPO.height / 2)
                    else:
                        avance = 62 if portador and portador.equipo == equipo else -28
                        p.objetivo = limitar_posicion(
                            p.casa[0] + signo * avance + (self.balon.x - CAMPO.centerx) * .12,
                            p.casa[1] + (self.balon.y - p.casa[1]) * .18)

        for p in self.jugadores:
            if p.control == "IA":
                self._mover_ia(p, dt)

    def _actualizar_posesion(self, dt: float) -> None:
        """Conducción, control de balones divididos y presión de la defensa IA."""
        if self.poseedor is not None:
            portador = self.poseedor
            self.balon.update(portador.x + math.cos(portador.angulo) * 19,
                              portador.y + math.sin(portador.angulo) * 19)
            self.altura = 0.0
            self.vel_balon.update(0, 0)
            rivales_ia = [p for p in self.jugadores
                          if p.control == "IA" and p.equipo != portador.equipo]
            for defensor in rivales_ia:
                if math.dist((defensor.x, defensor.y), (portador.x, portador.y)) < 23:
                    if defensor.recarga_robo <= 0 and portador.recarga_robo <= 0:
                        self.poseedor = defensor
                        defensor.recarga_robo = .8
                    break
            if self.poseedor is not None:
                return

        # La pelota se puede recuperar al llegar a ella; los pases en movimiento
        # siguen siendo interceptables si un jugador cruza su trayectoria.
        velocidad = self.vel_balon.length()
        receptor = min(self.jugadores,
                       key=lambda p: math.dist((p.x, p.y), self.balon), default=None)
        if receptor and math.dist((receptor.x, receptor.y), self.balon) < (22 if velocidad < 190 else 15):
            if self.altura < 24:
                self.poseedor = receptor
                self.vel_balon.update(0, 0)


    def _mover_ia(self, p: Jugador, dt: float) -> None:
        dx, dy = p.objetivo[0] - p.x, p.objetivo[1] - p.y
        distancia = math.hypot(dx, dy)
        if distancia > 3:
            velocidad = min(170, distancia * 3.2)
            p.x, p.y = limitar_posicion(p.x + dx / distancia * velocidad * dt,
                                         p.y + dy / distancia * velocidad * dt)
            p.angulo = math.atan2(dy, dx)

    def _patear(self, p: Jugador, accion: str, compañero: Jugador | None) -> None:
        if math.dist((p.x, p.y), self.balon) > 54:
            return
        if self.poseedor is not None and self.poseedor.equipo != p.equipo:
            return
        if accion == "pase" and compañero is None:
            return
        signo = 1 if p.equipo == 0 else -1
        if accion == "pase" and compañero:
            # Pase raso hacia el espacio al que corre el compañero.
            destino = pygame.Vector2(compañero.x + math.cos(compañero.angulo) * 34,
                                     compañero.y + math.sin(compañero.angulo) * 34)
            velocidad = 405
        elif accion == "centro":
            # Centro aéreo al área: busca un atacante y lo guía unos metros.
            meta_x = CAMPO.right - 150 if p.equipo == 0 else CAMPO.left + 150
            meta_y = compañero.y if compañero else CAMPO.centery
            destino = pygame.Vector2(meta_x, meta_y)
            velocidad = 440
            self.vel_altura = 205
        elif accion == "pase":
            # En un 1 contra 1, el pase se juega al espacio para avanzar la pelota.
            direccion = pygame.Vector2(math.cos(p.angulo), math.sin(p.angulo))
            destino = self.balon + direccion * 390
            velocidad = 390
        else:
            teclas = pygame.key.get_pressed()
            if p.control == "H1":
                dx = float(teclas[pygame.K_d]) - float(teclas[pygame.K_a])
                dy = float(teclas[pygame.K_s]) - float(teclas[pygame.K_w])
            elif p.control == "H2":
                dx = float(teclas[pygame.K_RIGHT]) - float(teclas[pygame.K_LEFT])
                dy = float(teclas[pygame.K_DOWN]) - float(teclas[pygame.K_UP])
            else:
                dx = dy = 0.0
            if math.hypot(dx, dy) > 0:
                p.angulo = math.atan2(dy, dx)
            elif p.control == "IA":
                p.angulo = 0.0 if signo == 1 else math.pi
            direccion = pygame.Vector2(math.cos(p.angulo), math.sin(p.angulo))
            destino = self.balon + direccion * 620
            velocidad = 570
            self.vel_altura = 155
        vector = destino - self.balon
        if vector.length_squared() == 0:
            vector.update(signo, 0)
        self.vel_balon = vector.normalize() * velocidad
        self.poseedor = None
        self.ultimo_golpeador = p
        self.audio.efecto("balon")
        self.audio.jugada(p.nombre, accion)
        self.ia.log.action(p.equipo, p.nombre, accion, self.tiempo)  # type: ignore[union-attr]
        self.numero_jugadas += 1

    def _actualizar_balon(self, dt: float) -> None:
        assert self.ia is not None
        if self.poseedor is not None:
            return
        self.vel_balon *= max(0, 1 - .62 * dt)
        self.balon += self.vel_balon * dt
        self.vel_altura -= 480 * dt
        self.altura = max(0, self.altura + self.vel_altura * dt)
        if self.altura == 0:
            self.vel_altura = max(0, -self.vel_altura * .43)
        if self.balon.y < CAMPO.top + 8 or self.balon.y > CAMPO.bottom - 8:
            self.balon.y = clamp(self.balon.y, CAMPO.top + 8, CAMPO.bottom - 8)
            self.vel_balon.y *= -.68
        if self.balon.x < CAMPO.left - 18 or self.balon.x > CAMPO.right + 18:
            if abs(self.balon.y - CAMPO.centery) < 72:
                equipo = 0 if self.balon.x > CAMPO.centerx else 1
                goleador = self.ultimo_golpeador
                self.ia.log.goal(equipo, goleador.nombre if goleador else "Equipo", self.tiempo)
                pais = obtener_seleccion(self.pais_indices[equipo]).nombre
                self.mensaje = f"¡GOOOL! · {pais} · {goleador.nombre if goleador else 'Equipo'}"
                self.mensaje_tiempo = 2.0
                self.audio.gol(goleador.nombre if goleador else "el equipo", pais)
                self.numero_jugadas += 1
                equipo_saque = 1 - equipo
                self._iniciar_saque(equipo_saque)
                self.mensaje += " · SAQUE " + ("AZUL" if equipo_saque == 0 else "ROJO")
                if max(self.ia.log.score.values()) >= 5:
                    self.fin = True
                    self.ia.log.finish()
            else:
                self.balon.x = clamp(self.balon.x, CAMPO.left - 18, CAMPO.right + 18)
                self.vel_balon.x *= -.68

    # ------------------------------- Dibujo ---------------------------------
    def dibujar(self) -> None:
        w, h = self.pantalla.get_size()
        self.pantalla.fill(FONDO)
        if self.estado == "menu":
            self._dibujar_menu(w, h)
            return
        self._dibujar_partido(w, h)

    def _texto(self, superficie: pygame.Surface, texto: str, xy: tuple[int, int],
               fuente: str = "normal", color: tuple[int, int, int] = BLANCO,
               centro: bool = False) -> pygame.Rect:
        imagen = self.fuentes[fuente].render(texto, True, color)
        rect = imagen.get_rect()
        if centro:
            rect.center = xy
        else:
            rect.topleft = xy
        superficie.blit(imagen, rect)
        return rect

    def _dibujar_menu(self, w: int, h: int) -> None:
        # Cabecera editorial, luces ambientales y panel de selección.
        pygame.draw.rect(self.pantalla, (12, 28, 45), (0, 0, w, h))
        pygame.draw.circle(self.pantalla, (18, 61, 72), (w - 60, 90), 230)
        pygame.draw.circle(self.pantalla, (19, 39, 65), (30, h - 20), 250)
        self._texto(self.pantalla, "PITCH / LAB", (90, 58), "bold", AMARILLO)
        self._texto(self.pantalla, "FÚTBOL 2D", (86, 102), "titulo")
        self._texto(self.pantalla, "Partido local · IA táctica · Estadísticas en vivo",
                    (92, 166), "normal", (167, 190, 204))
        panel = pygame.Rect(82, 204, min(w - 164, 720), 390)
        pygame.draw.rect(self.pantalla, (15, 29, 45), panel, border_radius=20)
        pygame.draw.rect(self.pantalla, (42, 66, 83), panel, 1, border_radius=20)
        self._texto(self.pantalla, "ELIGE TU PARTIDO", (panel.x + 28, panel.y + 22), "bold")
        descripciones = [
            ("UN JUGADOR", "Tú contra el equipo controlado por IA", AZUL),
            ("LOCAL", "Dos personas · un teclado", (118, 207, 166)),
            ("COOPERATIVO", "Una IA contra dos jugadores", AMARILLO),
            ("2 × 2", "Dos IA contra dos jugadores locales", ROJO),
        ]
        for i, (titulo, detalle, acento) in enumerate(descripciones):
            y = panel.y + 62 + i * 70
            tarjeta = pygame.Rect(panel.x + 22, y, panel.width - 44, 64)
            pygame.draw.rect(self.pantalla, (23, 43, 60), tarjeta, border_radius=12)
            pygame.draw.rect(self.pantalla, acento, (tarjeta.x, tarjeta.y, 5, tarjeta.height), border_radius=3)
            self._texto(self.pantalla, str(i + 1), (tarjeta.x + 22, tarjeta.y + 17), "bold", acento)
            self._texto(self.pantalla, titulo, (tarjeta.x + 66, tarjeta.y + 9), "bold")
            self._texto(self.pantalla, detalle, (tarjeta.x + 66, tarjeta.y + 36), "mini", (171, 191, 204))
            self._texto(self.pantalla, "PULSA " + str(i + 1), (tarjeta.right - 106, tarjeta.y + 24), "mini", acento)
        # Selector local de selecciones. A/D cambia el equipo azul; ←/→ el rojo.
        selector = pygame.Rect(832, 204, min(356, w - 862), 304)
        pygame.draw.rect(self.pantalla, (13, 29, 44), selector, border_radius=18)
        pygame.draw.rect(self.pantalla, (42, 66, 83), selector, 1, border_radius=18)
        self._texto(self.pantalla, "SELECCIONES NACIONALES", (selector.x + 20, selector.y + 19), "bold")
        for equipo in (0, 1):
            sel = obtener_seleccion(self.pais_indices[equipo])
            y = selector.y + 65 + equipo * 98
            tarjeta = pygame.Rect(selector.x + 14, y, selector.width - 28, 82)
            pygame.draw.rect(self.pantalla, (22, 43, 59), tarjeta, border_radius=12)
            self._texto(self.pantalla, "LOCAL" if equipo == 0 else "VISITANTE · IA / J2",
                        (tarjeta.x + 14, tarjeta.y + 8), "mini", (147, 172, 188))
            bandera = pygame.Rect(tarjeta.x + 15, tarjeta.y + 32, 50, 30)
            alto_franja = bandera.height // 3
            for franja, tono in enumerate(sel.bandera):
                pygame.draw.rect(self.pantalla, tono,
                                 (bandera.x, bandera.y + franja * alto_franja,
                                  bandera.width, alto_franja + 1))
            pygame.draw.rect(self.pantalla, (224, 234, 233), bandera, 1)
            self._texto(self.pantalla, f"{sel.nombre}  ·  {sel.sigla}",
                        (tarjeta.x + 78, tarjeta.y + 39), "bold", sel.camiseta)
        self._texto(self.pantalla, "11 TITULARES  ·  FORMACIÓN 4-3-3",
                    (selector.x + 20, selector.bottom + 19), "mini", AMARILLO)
        self._texto(self.pantalla, "A / D azul       ← / → rojo       R sortear selección de la IA",
                    (selector.x + 20, selector.bottom + 43), "mini", (171, 191, 204))
        self._texto(self.pantalla, "Centro: E (J1) / U (J2)  ·  audio y narración local",
                    (selector.x + 20, selector.bottom + 65), "mini", (145, 174, 188))
        previa = pygame.Rect(w - 285, 68, 245, 122)
        pygame.draw.rect(self.pantalla, (10, 23, 36), previa.inflate(10, 10), border_radius=10)
        pygame.draw.rect(self.pantalla, (80, 113, 128), previa.inflate(10, 10), 1, border_radius=10)
        if not self.recursos_visual.dibujar_vista_previa_cancha(self.pantalla, previa):
            pygame.draw.rect(self.pantalla, (30, 105, 62), previa, border_radius=3)
            pygame.draw.rect(self.pantalla, BLANCO, previa, 2)
            pygame.draw.line(self.pantalla, BLANCO, previa.midtop, previa.midbottom, 1)
            pygame.draw.circle(self.pantalla, BLANCO, previa.center, 18, 1)
        self._texto(self.pantalla, f"VISTA TÁCTICA · {ASSET_STYLE.upper()}",
                    (previa.centerx, previa.y - 10), "mini", AMARILLO, True)
        self.recursos_visual.dibujar_hinchas(self.pantalla, pygame.Rect(w - 445, 112, 140, 27))
        self._texto(self.pantalla, "J1 WASD · tiro ESPACIO · pase Q · centro E · robo F · cambio TAB · sprint SHIFT",
                    (92, h - 100), "normal", (192, 210, 219))
        self._texto(self.pantalla, "J2 FLECHAS · tiro CTRL DER. · pase ENTER · centro U · robo ALT · cambio BACKSPACE",
                    (92, h - 70), "mini", (146, 174, 190))
        self._texto(self.pantalla, "ESC salir", (w - 174, h - 48), "mini", (110, 140, 158))
        if self.recursos_visual.errores:
            self._texto(self.pantalla, "FALLBACK VECTORIAL ACTIVO", (w - 370, h - 22), "mini", AMARILLO)

    def _dibujar_partido(self, w: int, h: int) -> None:
        # Faja superior como marcador de transmisión deportiva.
        pygame.draw.rect(self.pantalla, (9, 20, 34), (0, 0, w, 114))
        pygame.draw.line(self.pantalla, (41, 72, 88), (0, 113), (w, 113), 1)
        self._texto(self.pantalla, "PITCH / LIVE", (34, 22), "bold", AMARILLO)
        posesion = (obtener_seleccion(self.pais_indices[self.poseedor.equipo]).sigla
                    if self.poseedor else "DIVIDIDA")
        nombre_poseedor = f" · {self.poseedor.nombre}" if self.poseedor else ""
        self._texto(self.pantalla, f"{MODOS[self.modo].upper()} · BALÓN {posesion}{nombre_poseedor}",
                    (36, 61), "mini", (155, 180, 195))
        assert self.ia is not None
        marcador = pygame.Rect(w // 2 - 190, 18, 380, 78)
        pygame.draw.rect(self.pantalla, (18, 38, 56), marcador, border_radius=15)
        pygame.draw.circle(self.pantalla, color_equipo(self.pais_indices[0]), (marcador.x + 45, marcador.centery), 8)
        pygame.draw.circle(self.pantalla, color_equipo(self.pais_indices[1]), (marcador.right - 45, marcador.centery), 8)
        self._texto(self.pantalla, obtener_seleccion(self.pais_indices[0]).sigla,
                    (marcador.x + 58, marcador.centery - 10), "mini", BLANCO)
        self._texto(self.pantalla, obtener_seleccion(self.pais_indices[1]).sigla,
                    (marcador.right - 94, marcador.centery - 10), "mini", BLANCO)
        self._texto(self.pantalla, str(self.ia.log.score[0]), (marcador.x + 83, marcador.centery), "titulo", color_equipo(self.pais_indices[0]), True)
        self._texto(self.pantalla, "—", (marcador.centerx, marcador.centery), "bold", (175, 193, 205), True)
        self._texto(self.pantalla, str(self.ia.log.score[1]), (marcador.right - 83, marcador.centery), "titulo", color_equipo(self.pais_indices[1]), True)
        restante = int(self.reloj_partido)
        self._texto(self.pantalla, f"{restante // 60:02}:{restante % 60:02}",
                    (w - 126, 34), "bold", AMARILLO)
        self._texto(self.pantalla, "P  pausa   ·   R  reiniciar   ·   ESC  pausa / menú",
                    (w - 320, 72), "mini", (150, 175, 190))
        self._dibujar_cancha()
        self._dibujar_entidades_campo()
        self._dibujar_pie(w, h)
        if self.pausa or self.fin:
            self._overlay(w, h, "PAUSA" if self.pausa else "PARTIDO FINALIZADO",
                          "Elige continuar, reiniciar o volver al menú" if self.pausa
                          else "Elige una revancha o vuelve a la selección")

    def _dibujar_cancha(self) -> None:
        # Gradas y césped con franjas; la geometría está calculada en píxeles de pantalla.
        grada = CAMPO.inflate(38, 34)
        pygame.draw.rect(self.pantalla, (24, 38, 49), grada, border_radius=8)
        pygame.draw.rect(self.pantalla, (67, 83, 82), CAMPO, border_radius=3)
        banda = CAMPO.width / 12
        for i in range(12):
            x = round(CAMPO.left + i * banda)
            pygame.draw.rect(self.pantalla, VERDE_1 if i % 2 == 0 else VERDE_2,
                             (x, CAMPO.top, math.ceil(banda), CAMPO.height))
        pygame.draw.rect(self.pantalla, BLANCO, CAMPO, 3)
        pygame.draw.line(self.pantalla, BLANCO, CAMPO.midtop, CAMPO.midbottom, 2)
        pygame.draw.circle(self.pantalla, BLANCO, CAMPO.center, 78, 2)
        pygame.draw.circle(self.pantalla, BLANCO, CAMPO.center, 4)
        area_alto, area_ancho = 214, 300
        for lado in (0, 1):
            x = CAMPO.left if lado == 0 else CAMPO.right - area_alto
            pygame.draw.rect(self.pantalla, BLANCO,
                             (x, CAMPO.centery - area_ancho // 2, area_alto, area_ancho), 2)
            x_chica = CAMPO.left if lado == 0 else CAMPO.right - 78
            pygame.draw.rect(self.pantalla, BLANCO,
                             (x_chica, CAMPO.centery - 116, 78, 232), 2)
            punto_x = CAMPO.left + 144 if lado == 0 else CAMPO.right - 144
            pygame.draw.circle(self.pantalla, BLANCO, (punto_x, CAMPO.centery), 4)
            pygame.draw.arc(self.pantalla, BLANCO,
                            (punto_x - 58, CAMPO.centery - 58, 116, 116),
                            -1.1 if lado else 2.05, 1.1 if lado else 4.23, 2)
            # Arco de esquina y marco con red.
            gx = CAMPO.left - 14 if lado == 0 else CAMPO.right
            gy = CAMPO.centery - 72
            porteria = pygame.Rect(gx, gy, 14, 144)
            pygame.draw.rect(self.pantalla, (221, 233, 234), porteria, 2)
            for yy in range(gy + 8, gy + 144, 12):
                pygame.draw.line(self.pantalla, (122, 159, 160), (gx, yy), (gx + 14, yy), 1)
            for xx in range(gx, gx + 15, 7):
                pygame.draw.line(self.pantalla, (122, 159, 160), (xx, gy), (xx, gy + 144), 1)
        for y, hacia in ((CAMPO.top, 1), (CAMPO.bottom, -1)):
            pygame.draw.arc(self.pantalla, BLANCO,
                            (CAMPO.left - 12, y - 12 if hacia == 1 else y - 12, 24, 24),
                            0 if hacia == 1 else math.pi, math.pi if hacia == 1 else 2 * math.pi, 2)

    def _dibujar_entidades_campo(self) -> None:
        """Ordena jugadores y balón por profundidad de pantalla (eje Y)."""
        entidades: list[tuple[float, int, Jugador | None]] = [
            (float(p.y), 0, p) for p in self.jugadores
        ]
        # A igual profundidad, el balón se pinta encima para conservar legibilidad.
        entidades.append((float(self.balon.y), 1, None))
        entidades.sort(key=lambda entidad: (entidad[0], entidad[1]))
        for _, tipo, jugador in entidades:
            if tipo == 1:
                self._dibujar_balon()
            else:
                assert jugador is not None
                self._dibujar_jugador(jugador)

    def _dibujar_jugador(self, p: Jugador) -> None:
        color_uniforme = color_equipo(self.pais_indices[p.equipo])
        x, y = round(p.x), round(p.y)
        if self.recursos_visual.tiene_jugadores(p.equipo):
            if self.poseedor is p:
                pygame.draw.circle(self.pantalla, AMARILLO, (x, y), 24, 2)
            elif p.control in ("H1", "H2"):
                color_control = (111, 222, 255) if p.control == "H1" else (255, 221, 121)
                pygame.draw.circle(self.pantalla, color_control, (x, y), 22, 2)
            if self.recursos_visual.dibujar_jugador(self.pantalla, p.equipo, p.indice,
                                                    (x, y), p.angulo, color_uniforme):
                if p.control == "IA":
                    pygame.draw.circle(self.pantalla, (240, 240, 226), (x, y - 25), 2)
                return
        # Fallback vectorial con orientación real y sombra cuando falta una imagen.
        if self.poseedor is p:
            pygame.draw.circle(self.pantalla, AMARILLO, (x, y), 22, 2)
        elif p.control in ("H1", "H2"):
            color_control = (111, 222, 255) if p.control == "H1" else (255, 221, 121)
            pygame.draw.circle(self.pantalla, color_control, (x, y), 20, 2)
        pygame.draw.ellipse(self.pantalla, (13, 43, 34), (x - 16, y + 11, 34, 11))
        ux, uy = math.cos(p.angulo), math.sin(p.angulo)
        perpendicular = pygame.Vector2(-uy, ux)
        centro = pygame.Vector2(x, y)
        frente = pygame.Vector2(ux, uy)
        hombro = centro + frente * 1
        piel = (194, 145, 111)
        for lado in (-1, 1):
            pierna = centro - frente * 6 + perpendicular * (lado * 5)
            pie = pierna - frente * 8
            pygame.draw.line(self.pantalla, (31, 38, 43), pierna, pie, 5)
            pygame.draw.line(self.pantalla, (235, 239, 232), pie,
                             pie - frente * 5, 3)
            brazo = hombro + perpendicular * (lado * 8)
            mano = brazo + frente * 2 + perpendicular * (lado * 5)
            pygame.draw.line(self.pantalla, color_uniforme, brazo, mano, 5)
            pygame.draw.circle(self.pantalla, piel, (round(mano.x), round(mano.y)), 2)
        pantalon = centro - frente * 7
        pygame.draw.ellipse(self.pantalla, (30, 38, 48),
                            (round(pantalon.x - 8), round(pantalon.y - 5), 16, 12))
        vertices = [centro + frente * 10 - perpendicular * 7,
                    centro + frente * 9 + perpendicular * 7,
                    centro - frente * 7 + perpendicular * 6,
                    centro - frente * 8 - perpendicular * 6]
        puntos = [(round(v.x), round(v.y)) for v in vertices]
        pygame.draw.polygon(self.pantalla, (17, 30, 43), puntos)
        pygame.draw.polygon(self.pantalla, color_uniforme, puntos)
        pygame.draw.line(self.pantalla, (239, 242, 234), centro - frente * 5,
                         centro + frente * 6, 2)
        cabeza = centro + frente * 13
        pygame.draw.circle(self.pantalla, (85, 59, 48),
                           (round(cabeza.x), round(cabeza.y)), 5)
        pygame.draw.circle(self.pantalla, piel,
                           (round(cabeza.x), round(cabeza.y + 1)), 4)
        numero = str(p.indice + 1)
        img = self.fuentes["numero"].render(numero, True, (255, 255, 255))
        self.pantalla.blit(img, img.get_rect(center=(round(x + ux), round(y + uy))))
        punta = (round(x + ux * 22), round(y + uy * 22))
        pygame.draw.circle(self.pantalla, AMARILLO, punta, 3)
        if p.control == "IA":
            pygame.draw.circle(self.pantalla, (240, 240, 226), (x, y - 21), 2)

    def _dibujar_balon(self) -> None:
        x, y = round(self.balon.x), round(self.balon.y - self.altura * .45)
        radio_sombra = max(4, 9 - round(self.altura * .035))
        pygame.draw.ellipse(self.pantalla, (14, 44, 32),
                            (round(self.balon.x - radio_sombra), round(self.balon.y + 6),
                             radio_sombra * 2, max(4, radio_sombra)))
        if self.recursos_visual.dibujar_balon(self.pantalla, (x, y)):
            return
        pygame.draw.circle(self.pantalla, (251, 251, 238), (x, y), 10)
        pygame.draw.circle(self.pantalla, (35, 42, 50), (x, y), 10, 1)
        for i in range(5):
            ang = i * (2 * math.pi / 5) - math.pi / 2
            px, py = round(x + math.cos(ang) * 4), round(y + math.sin(ang) * 4)
            pygame.draw.circle(self.pantalla, (39, 47, 54), (px, py), 2)

    def _dibujar_pie(self, w: int, h: int) -> None:
        pygame.draw.rect(self.pantalla, (9, 20, 34), (0, h - 82, w, 82))
        pygame.draw.line(self.pantalla, (41, 72, 88), (0, h - 82), (w, h - 82), 1)
        controles = "J1  WASD  ·  ESPACIO tiro  ·  Q pase  ·  E centro  ·  F robo  ·  TAB cambia  ·  SHIFT corre"
        self._texto(self.pantalla, controles, (34, h - 58), "mini", (189, 205, 215))
        self._texto(self.pantalla, "J2 FLECHAS · CTRL DER. tiro · ENTER pase · U centro · ALT robo · BKSP cambia · SHIFT DER. corre",
                    (34, h - 34), "mini", (145, 172, 189))
        self._texto(self.pantalla, f"ACCIONES REGISTRADAS  {self.numero_jugadas:03}",
                    (w - 286, h - 48), "mini", (117, 151, 172))
        if self.recursos_visual.errores:
            self._texto(self.pantalla, "FALLBACK VECTORIAL ACTIVO", (34, h - 14), "mini", AMARILLO)
        if self.mensaje_tiempo > 0 and self.mensaje:
            panel = pygame.Surface((440, 54), pygame.SRCALPHA)
            panel.fill((7, 17, 29, 210))
            self.pantalla.blit(panel, (w // 2 - 220, 124))
            self._texto(self.pantalla, self.mensaje, (w // 2, 151), "bold", AMARILLO, True)

    def _overlay(self, w: int, h: int, titulo: str, subtitulo: str) -> None:
        self.botones_overlay.clear()
        capa = pygame.Surface((w, h), pygame.SRCALPHA)
        capa.fill((3, 10, 18, 188))
        self.pantalla.blit(capa, (0, 0))
        caja = pygame.Rect(w // 2 - 270, h // 2 - 166, 540, 332)
        pygame.draw.rect(self.pantalla, (12, 28, 44), caja, border_radius=20)
        pygame.draw.rect(self.pantalla, (74, 104, 120), caja, 1, border_radius=20)
        self._texto(self.pantalla, titulo, (w // 2, h // 2 - 126), "titulo", AMARILLO, True)
        self._texto(self.pantalla, subtitulo, (w // 2, h // 2 - 83), "normal", BLANCO, True)
        if self.fin:
            botones = [("jugar_de_nuevo", "REINICIAR PARTIDO"),
                       ("menu", "VOLVER AL MENÚ"), ("salir", "SALIR DEL JUEGO")]
        else:
            botones = [("continuar", "CONTINUAR JUGANDO"),
                       ("reiniciar", "REINICIAR PARTIDO"), ("menu", "VOLVER AL MENÚ")]
        mouse = pygame.mouse.get_pos()
        for i, (accion, etiqueta) in enumerate(botones):
            rect = pygame.Rect(w // 2 - 205, h // 2 - 48 + i * 61, 410, 48)
            tono = (37, 89, 113) if rect.collidepoint(mouse) else (22, 46, 64)
            pygame.draw.rect(self.pantalla, tono, rect, border_radius=10)
            pygame.draw.rect(self.pantalla, (62, 117, 138), rect, 1, border_radius=10)
            self._texto(self.pantalla, etiqueta, rect.center, "normal", BLANCO, True)
            self.botones_overlay[accion] = rect


if __name__ == "__main__":
    Futbol2D().bucle()
