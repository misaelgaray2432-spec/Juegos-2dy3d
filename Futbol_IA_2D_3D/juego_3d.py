"""Fútbol 3D volumétrico con IA, estadio, física y controles locales."""
from __future__ import annotations

import argparse
import math
import random
import sys
from typing import TYPE_CHECKING, Any, cast
from pathlib import Path

# IA y dependencias compatibles con Windows se cargan antes de Panda/Ursina.
from datetime import datetime
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

FORMACION_7 = (
    (.045, 0.0),
    (.22, -.68), (.22, .68),
    (.43, -.58), (.43, .58),
    (.735, -.40), (.735, .40),
)


def obtener_seleccion(indice: int) -> Seleccion:
    return SELECCIONES[indice % len(SELECCIONES)]


def formacion(equipo: int, indice: int, x_min: float, x_max: float,
              y_centro: float, y_radio: float) -> tuple[float, float]:
    """Convierte la formación 4-3-3 a coordenadas; ambos atacan hacia el centro rival."""
    avance, carril = FORMACION_433[indice % 11]
    x = x_min + avance * (x_max - x_min) if equipo == 0 else x_max - avance * (x_max - x_min)
    return x, y_centro + carril * y_radio


def formacion_7(equipo: int, indice: int, x_min: float, x_max: float,
                z_centro: float, z_radio: float) -> tuple[float, float]:
    """Distribuye siete jugadores en 1-2-2-2 sobre el campo reducido."""
    avance, carril = FORMACION_7[indice % len(FORMACION_7)]
    x = x_min + avance * (x_max - x_min) if equipo == 0 else x_max - avance * (x_max - x_min)
    return x, z_centro + carril * z_radio


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

import pygame

if TYPE_CHECKING:
    from ursina.texture import Texture as UrsinaTexture
else:
    UrsinaTexture = Any


# Estilo de recursos para esta escena. Cambiarlo a "2d" prueba la variante
# ilustrada del mismo paquete, cargando solo una de las dos carpetas.
ASSET_STYLE = "3d"
if ASSET_STYLE not in {"2d", "3d"}:
    raise ValueError('ASSET_STYLE debe ser "2d" o "3d".')

# El juego ofrece ambos escenarios. El estadio profesional queda activo;
# "barrio" selecciona la variante comunitaria con casas y malla.
ESCENARIO_3D = "estadio"
if ESCENARIO_3D not in {"barrio", "estadio"}:
    raise ValueError('ESCENARIO_3D debe ser "barrio" o "estadio".')

class RecursosVisuales3D:
    """Carga recursos de un solo estilo y conserva alternativas del juego."""

    ARCHIVOS = {
        "cancha": "cancha/cancha_completa.png",
        "cesped": "cancha/cesped_estadio.png" if ASSET_STYLE == "3d" else None,
        "estadio": "estadios/estadio_01.png",
        "hinchas_1": "hinchas/hinchas_fila_1.png",
        "hinchas_2": "hinchas/hinchas_fila_2.png",
        "jugadores_0": "jugadores/jugadores_amarillo_azul.png",
        "jugadores_1": "jugadores/jugadores_rojo_blanco.png",
    }

    def __init__(self) -> None:
        self.raiz = Path(__file__).resolve().parent / f"assets_{ASSET_STYLE}"
        self._texturas: dict[str, UrsinaTexture | None] = {}
        self._hojas: dict[
            tuple[int, tuple[int, int, int]], list[UrsinaTexture] | None
        ] = {}
        self._porteros: dict[int, UrsinaTexture | None] = {}
        self.errores: list[str] = []

    def textura(self, nombre: str) -> UrsinaTexture | None:
        """Carga una textura por su nombre lógico; devuelve None ante un fallo."""
        if nombre in self._texturas:
            return self._texturas[nombre]
        relativo = self.ARCHIVOS.get(nombre)
        if relativo is None and nombre in self.ARCHIVOS:
            self._texturas[nombre] = None
            return None
        if relativo is None:
            self.errores.append(f"Recurso 3D desconocido: {nombre}")
            return None
        ruta = self.raiz / relativo
        try:
            if not ruta.is_file():
                raise FileNotFoundError(ruta)
            # Carga la ruta absoluta del proyecto, aunque el juego se inicie
            # desde otra carpeta.
            from PIL import Image
            from ursina import Texture

            with Image.open(ruta) as imagen_fuente:
                imagen = imagen_fuente.convert("RGBA")
            textura = Texture(imagen, filtering="bilinear")
            textura.repeat_setter(False)
            self._texturas[nombre] = textura
            return textura
        except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
            self.errores.append(f"{ruta}: {error}")
            self._texturas[nombre] = None
            return None

    def textura_portero(self, equipo: int) -> UrsinaTexture | None:
        """Extrae el portero con alfa real desde los componentes verificables de la hoja."""
        if equipo in self._porteros:
            return self._porteros[equipo]
        ruta = self.raiz / "jugadores" / "porteros_y_arbitros.png"
        try:
            if equipo not in (0, 1):
                raise ValueError(f"Equipo de portero no válido: {equipo}")
            if not ruta.is_file():
                raise FileNotFoundError(ruta)
            from PIL import Image, ImageChops
            from ursina import Texture

            superficie = pygame.image.load(str(ruta))
            mascara = pygame.mask.from_surface(superficie, 96)
            componentes = []
            for componente in mascara.connected_components(minimum=1000):
                rectangulos = cast(list[pygame.Rect], componente.get_bounding_rects())
                if not rectangulos:
                    continue
                caja = rectangulos[0]
                if caja.width >= 45 and caja.height >= 70:
                    componentes.append((caja, componente))
            componentes.sort(key=lambda elemento: elemento[0].x)
            # El primer recorte es el portero verde y el segundo el azul.
            # Se aplica la máscara del componente para no incluir jugadores vecinos.
            candidatos = [elemento for elemento in componentes if elemento[0].x < 205]
            if len(candidatos) < 2:
                raise ValueError("No se detectaron los dos porteros al inicio de la hoja")
            caja, componente = candidatos[equipo]
            superficie_mascara = componente.to_surface(
                setcolor=(255, 255, 255, 255), unsetcolor=(0, 0, 0, 0)
            )
            mascara_pil = Image.frombytes(
                "RGBA", superficie_mascara.get_size(),
                pygame.image.tostring(superficie_mascara, "RGBA")
            ).getchannel("A").crop((caja.left, caja.top, caja.right, caja.bottom))
            imagen = Image.open(ruta).convert("RGBA").crop(
                (caja.left, caja.top, caja.right, caja.bottom)
            )
            imagen.putalpha(ImageChops.multiply(imagen.getchannel("A"), mascara_pil))
            textura = Texture(imagen, filtering="bilinear")
            textura.repeat_setter(False)
            self._porteros[equipo] = textura
            return textura
        except (FileNotFoundError, OSError, RuntimeError, ValueError, ImportError,
                pygame.error) as error:
            self.errores.append(f"{ruta}: no se pudo recortar el portero ({error})")
            self._porteros[equipo] = None
            return None

    def hoja_jugadores(
        self, equipo: int, camiseta: tuple[int, int, int]
    ) -> list[UrsinaTexture] | None:
        """Recolorea y recorta cuerpos conectados sin asumir celdas uniformes."""
        clave = (equipo, camiseta)
        if clave in self._hojas:
            return self._hojas[clave]
        nombre = f"jugadores_{equipo}"
        ruta = self.raiz / self.ARCHIVOS[nombre]
        try:
            from PIL import Image, ImageChops
            from ursina import Texture

            superficie = pygame.image.load(str(ruta))
            mascara = pygame.mask.from_surface(superficie, 96)
            candidatos = []
            for componente in mascara.connected_components(minimum=1000):
                rectangulos = cast(list[pygame.Rect], componente.get_bounding_rects())
                if not rectangulos:
                    continue
                caja = rectangulos[0]
                if (
                    caja.height >= 80 and 30 <= caja.width <= 100
                    and caja.y < superficie.get_height() * .22
                ):
                    candidatos.append((caja, componente))
            candidatos.sort(key=lambda item: item[0].x)
            carrera = [
                item for item in candidatos
                if item[0].x < superficie.get_width() * .76
            ]
            celebracion = [
                item for item in candidatos
                if item[0].x >= superficie.get_width() * .76
            ]

            if len(carrera) < 6:
                raise ValueError(
                    f"se detectaron {len(carrera)} cuadros de carrera; se necesitan 6"
                )

            imagen = Image.open(ruta).convert("RGBA")
            objetivo = colorsys.rgb_to_hsv(*(canal / 255 for canal in camiseta))
            pixeles = imagen.load()
            if pixeles is None:
                raise RuntimeError("No se pudieron cargar los píxeles de la hoja")
            es_amarilla = equipo == 0
            for y in range(imagen.height):
                for x in range(imagen.width):
                    rojo, verde, azul, alfa = cast(
                        tuple[int, int, int, int], pixeles[x, y]
                    )
                    if not alfa:
                        continue
                    tono, saturacion, valor = colorsys.rgb_to_hsv(
                        rojo / 255, verde / 255, azul / 255)
                    camiseta_base = (
                        0.075 <= tono <= 0.20 if es_amarilla
                        else tono <= 0.055 or tono >= 0.945
                    )
                    if camiseta_base and saturacion > 0.30:
                        nuevo = colorsys.hsv_to_rgb(
                            objetivo[0],
                            min(1.0, max(0.42, objetivo[1] * (0.58 + 0.42 * saturacion))),
                            min(1.0, valor * (0.72 + 0.28 * objetivo[2]) / 0.92),
                        )
                        pixeles[x, y] = (
                            round(nuevo[0] * 255), round(nuevo[1] * 255),
                            round(nuevo[2] * 255), alfa,
                        )

            texturas = []
            recortes = carrera[:6] + celebracion[-1:]
            for caja, componente in recortes:
                mascara_superficie = componente.to_surface(
                    setcolor=(255, 255, 255, 255), unsetcolor=(0, 0, 0, 0)
                )
                mascara_pil = Image.frombytes(
                    "RGBA", mascara_superficie.get_size(),
                    pygame.image.tostring(mascara_superficie, "RGBA")
                ).getchannel("A")
                recorte = caja.inflate(4, 4).clip(superficie.get_rect())
                limites = (recorte.left, recorte.top, recorte.right, recorte.bottom)
                figura = imagen.crop(limites)
                figura.putalpha(ImageChops.multiply(
                    figura.getchannel("A"), mascara_pil.crop(limites)
                ))
                textura = Texture(figura, filtering="bilinear")
                textura.repeat_setter(False)
                texturas.append(textura)

            self._hojas[clave] = texturas
            return texturas
        except (FileNotFoundError, OSError, RuntimeError, ValueError,
                ImportError, pygame.error) as error:
            self.errores.append(f"{ruta}: no se pudieron recortar jugadores ({error})")
            self._hojas[clave] = None
            return None


import math


class AnimadorProcedural:
    """Anima articulaciones existentes sin crear geometría en cada fotograma."""

    DURACIONES = {
        "giro": .20,
        "pase": .30,
        "centro": .38,
        "disparo": .46,
        "celebracion": 1.70,
        "portero_salto": .62,
        "portero_estirada": .62,
    }

    def __init__(self, jugador: dict) -> None:
        self.tiempo_restante = 0.0
        self.direccion = 1
        self.ultimo_yaw = float(jugador["e"].rotation_y)
        self.estado = "espera"
        jugador["estado_animacion"] = "espera"

    def activar(
        self, estado: str, duracion: float | None = None, direccion: int = 1
    ) -> None:
        """Inicia una animación con duración limitada."""
        if duracion is None:
            duracion = self.DURACIONES.get(estado, .25)
        self.tiempo_restante = max(0.0, float(duracion))
        self.direccion = 1 if direccion >= 0 else -1
        self.estado = estado

    @staticmethod
    def _limpiar_pose(jugador: dict) -> None:
        jugador["cuerpo"].rotation_x = 0
        jugador["cuerpo"].rotation_y = 0
        jugador["cuerpo"].rotation_z = 0
        for pierna in jugador["piernas"]:
            pierna.rotation_z = 0
        for indice, brazo in enumerate(jugador["brazos"]):
            brazo.rotation_z = -12 if indice == 0 else 12
        if jugador["e"].y > .04:
            jugador["e"].y = 0

    def actualizar(self, jugador: dict, dt: float) -> None:
        """Actualiza la pose según movimiento, acción o función del portero."""
        self.tiempo_restante = max(0.0, self.tiempo_restante - dt)
        yaw = float(jugador["e"].rotation_y)
        diferencia = abs((yaw - self.ultimo_yaw + 180) % 360 - 180)
        self.ultimo_yaw = yaw

        if self.tiempo_restante > 0:
            estado = self.estado
        elif diferencia > 42:
            estado = "giro"
            self.tiempo_restante = .12
            self.estado = estado
        else:
            estado = (
                "carrera" if jugador.get("velocidad_anim", 0.0) > .16
                else "espera"
            )
        jugador["estado_animacion"] = estado
        self._limpiar_pose(jugador)

        velocidad = float(jugador.get("velocidad_anim", 0.0))
        fase = float(jugador.get("fase", 0.0))
        paso = min(39.0, 13.0 + velocidad * 3.1)
        swing = math.sin(fase) * paso

        if estado == "carrera":
            jugador["piernas"][0].rotation_z = swing
            jugador["piernas"][1].rotation_z = -swing
            jugador["brazos"][0].rotation_z = -12 - swing * .52
            jugador["brazos"][1].rotation_z = 12 + swing * .52
        elif estado == "giro":
            jugador["cuerpo"].rotation_z = -self.direccion * 12
            jugador["brazos"][0].rotation_z = -28
            jugador["brazos"][1].rotation_z = 28
        elif estado in ("pase", "centro", "disparo"):
            potencia = 33 if estado == "pase" else 48 if estado == "centro" else 72
            jugador["piernas"][1].rotation_z = -potencia
            jugador["brazos"][0].rotation_z = -32
            jugador["brazos"][1].rotation_z = 42
            jugador["cuerpo"].rotation_x = -8 if estado != "disparo" else -15
        elif estado == "celebracion":
            jugador["brazos"][0].rotation_z = 150
            jugador["brazos"][1].rotation_z = 210
            jugador["piernas"][0].rotation_z = -13
            jugador["piernas"][1].rotation_z = 13
            jugador["e"].y = .04 + abs(math.sin(fase * 1.5)) * .12
        elif estado in ("portero_salto", "portero_estirada"):
            inclinacion = 54 if estado == "portero_estirada" else 38
            jugador["cuerpo"].rotation_z = -self.direccion * inclinacion
            jugador["brazos"][0].rotation_z = -82
            jugador["brazos"][1].rotation_z = 82
            jugador["piernas"][0].rotation_z = 25 * self.direccion
            jugador["piernas"][1].rotation_z = -25 * self.direccion
            if estado == "portero_salto":
                jugador["e"].y = .08 + abs(math.sin(fase * 2)) * .17


import random


def construir_detalle_estadio(parent, *, Entity, Cylinder, color) -> dict[str, int]:
    """Construye gradas por sectores, con asientos, pasillos e hinchada ligera."""
    rng = random.Random(41)
    paleta_asientos = [
        color.rgb32(28, 68, 99), color.rgb32(37, 88, 124),
        color.rgb32(49, 104, 139), color.rgb32(104, 48, 62),
    ]
    paleta_publico = [
        color.rgb32(45, 135, 213), color.rgb32(224, 70, 75),
        color.rgb32(245, 194, 57), color.rgb32(231, 235, 226),
        color.rgb32(43, 158, 111), color.rgb32(117, 83, 185),
        color.rgb32(224, 120, 48),
    ]
    paleta_piel = [
        color.rgb32(225, 179, 142), color.rgb32(191, 139, 103),
        color.rgb32(151, 103, 76), color.rgb32(120, 77, 56),
    ]
    asientos = 0
    espectadores = 0

    def espectador(posicion, color_camiseta, animado=False):
        nonlocal espectadores
        x, y, z = posicion
        tono_piel = rng.choice(paleta_piel)
        Entity(parent=parent, model="cube", color=color_camiseta,
               position=Vec3(x, y + .30, z), scale=Vec3(.19, .34, .16),
               unlit=True)
        Entity(parent=parent, model="sphere", color=tono_piel,
               position=Vec3(x, y + .55, z),
               scale=Vec3(.15, .15, .15), unlit=True)
        if animado:
            brazo = Entity(parent=parent, model=Cylinder(), color=color_camiseta,
                           position=Vec3(x - .14, y + .45, z),
                           scale=Vec3(.055, .34, .055), rotation_z=-31,
                           unlit=True)
            Entity(parent=parent, model="sphere", color=tono_piel,
                   position=Vec3(brazo.x - .09, y + .62, z),
                   scale=Vec3(.07, .07, .07),
                   unlit=True)
        espectadores += 1

    def asiento_lateral(x, y, z, lado):
        nonlocal asientos
        tono = rng.choice(paleta_asientos)
        Entity(parent=parent, model="cube", color=tono,
               position=Vec3(x, y + .055, z), scale=Vec3(.79, .13, .50),
               unlit=True)
        Entity(parent=parent, model="cube", color=tono,
               position=Vec3(x, y + .27, z + lado * .20),
               scale=Vec3(.76, .34, .075), unlit=True)
        asientos += 1

    def asiento_fondo(x, y, z, lado):
        nonlocal asientos
        tono = rng.choice(paleta_asientos)
        Entity(parent=parent, model="cube", color=tono,
               position=Vec3(x, y + .055, z), scale=Vec3(.50, .13, .79),
               unlit=True)
        Entity(parent=parent, model="cube", color=tono,
               position=Vec3(x + lado * .20, y + .27, z),
               scale=Vec3(.075, .34, .76), unlit=True)
        asientos += 1

    # Tres bandejas escalonadas dejan un pasillo claro entre el césped y las
    # localidades; las butacas se separan en cuatro sectores de acceso.
    for lado in (-1, 1):
        for nivel in range(4):
            altura = .47 + nivel * .88
            z = lado * (8.63 + nivel * 1.40)
            tono_grada = color.rgb32(35 + nivel * 5, 54 + nivel * 6, 72 + nivel * 7)
            Entity(parent=parent, model="cube", color=tono_grada,
                   position=Vec3(0, altura - .15, z),
                   scale=Vec3(29.2, .30, 1.24), unlit=True)
            Entity(parent=parent, model="cube",
                   color=color.rgb32(19, 37, 53),
                   position=Vec3(0, altura - .33, z - lado * .53),
                   scale=Vec3(29.2, .16, .12), unlit=True)

            for indice in range(24):
                x = -13.15 + indice * 1.145
                if any(abs(x - pasillo) < .53 for pasillo in (-7.15, 0, 7.15)):
                    continue
                z_asiento = lado * (8.30 + nivel * 1.40)
                asiento_lateral(x, altura, z_asiento, lado)
                if rng.random() > .08:
                    tono = rng.choice(paleta_publico)
                    espectador((x, altura + .04, z_asiento),
                               tono, animado=(indice + nivel) % 7 == 0)

            # Escaleras transversales y barandillas dividen la tribuna en
            # sectores visibles desde la cámara.
            for pasillo in (-7.15, 0, 7.15):
                Entity(parent=parent, model="cube", color=color.rgb32(136, 151, 155),
                       position=Vec3(pasillo, altura + .02, z - lado * .34),
                       scale=Vec3(.09, .05, .54), unlit=True)
                Entity(parent=parent, model=Cylinder(),
                       color=color.rgb32(148, 165, 169),
                       position=Vec3(pasillo - .39, altura + .31, z),
                       scale=Vec3(.035, .56, .035), unlit=True)
                Entity(parent=parent, model=Cylinder(),
                       color=color.rgb32(148, 165, 169),
                       position=Vec3(pasillo + .39, altura + .31, z),
                       scale=Vec3(.035, .56, .035), unlit=True)

    # Fondos de tribuna tras las porterías, con asientos orientados hacia el
    # campo y filas de hinchas en los mismos sectores escalonados.
    for lado in (-1, 1):
        for nivel in range(4):
            altura = .47 + nivel * .88
            x = lado * (14.70 + nivel * 1.37)
            tono_grada = color.rgb32(35 + nivel * 5, 54 + nivel * 6, 72 + nivel * 7)
            Entity(parent=parent, model="cube", color=tono_grada,
                   position=Vec3(x, altura - .15, 0),
                   scale=Vec3(1.22, .30, 14.8), unlit=True)
            Entity(parent=parent, model="cube", color=color.rgb32(19, 37, 53),
                   position=Vec3(x - lado * .53, altura - .33, 0),
                   scale=Vec3(.12, .16, 14.8), unlit=True)
            for indice in range(12):
                z = -6.30 + indice * 1.145
                if any(abs(z - pasillo) < .53 for pasillo in (-3.45, 3.45)):
                    continue
                x_asiento = lado * (14.35 + nivel * 1.37)
                asiento_fondo(x_asiento, altura, z, lado)
                if rng.random() > .10:
                    tono = rng.choice(paleta_publico)
                    espectador((x_asiento, altura + .04, z),
                               tono, animado=(indice + nivel) % 6 == 0)

    # Banderas pequeñas sobre la barandilla y pancartas de sector aportan
    # color sin introducir texturas ni marcas externas.
    for lado in (-1, 1):
        for indice, x in enumerate((-11.8, -4.0, 4.0, 11.8)):
            poste_x = x
            poste_z = lado * 8.05
            Entity(parent=parent, model=Cylinder(), color=color.rgb32(215, 223, 218),
                   position=Vec3(poste_x, 1.14, poste_z), scale=Vec3(.025, .92, .025),
                   unlit=True)
            tonos = (
                color.rgb32(39, 137, 196), color.rgb32(235, 202, 54),
                color.rgb32(200, 54, 61),
            )
            for franja, tono in enumerate(tonos):
                Entity(parent=parent, model="cube", color=tono,
                       position=Vec3(poste_x + .17 + franja * .03, 1.40 - franja * .12, poste_z),
                       scale=Vec3(.34, .18, .025), unlit=True)

    return {"asientos": asientos, "espectadores_3d": espectadores}

from ursina import (
    AmbientLight, Button, Cylinder, DirectionalLight, Entity, Func, Mesh, Text,
    destroy, Ursina, Vec2, Vec3, application, camera, color, held_keys, invoke,
    time, window,
)
from ursina.shaders import lit_with_shadows_shader

# ===== Motor del partido 3D =====

LARGO, ANCHO = 24.0, 14.0
VERDE_1 = color.rgb32(37, 132, 64)
VERDE_2 = color.rgb32(49, 153, 75)
BLANCO = color.rgb32(239, 246, 238)
AZUL = color.rgb32(31, 139, 255)
ROJO = color.rgb32(244, 71, 81)
AMARILLO = color.rgb32(255, 203, 66)


def segmento(a: tuple[float, float], b: tuple[float, float], parent: Entity,
             grosor: float = .045, y: float = .012, tono=BLANCO) -> Entity:
    """Dibuja una línea sobre el plano XZ usando una caja fina."""
    dx, dz = b[0] - a[0], b[1] - a[1]
    largo = math.hypot(dx, dz)
    e = Entity(model="cube", parent=parent, color=tono,
               position=Vec3((a[0] + b[0]) / 2, y, (a[1] + b[1]) / 2),
               scale=Vec3(largo, .025, grosor))
    e.rotation_y = -math.degrees(math.atan2(dz, dx))
    return e


class Juego3D(Entity):
    """Partido 3D con equipos mixtos, físicas sencillas y HUD deportivo."""

    def __init__(self) -> None:
        super().__init__()
        window.color = color.rgb32(11, 26, 42)
        window.exit_button.visible = False
        camera.position = Vec3(0, 24, -30)
        camera.rotation = Vec3(38, 0, 0)
        camera.fov = 61
        self.estado = "menu"
        self.modo = 1
        self.pais_indices = [0, 1]
        self.pais_etiquetas: list[Text] = []
        self.pais_escudos: list[Entity] = []
        self.audio = AudioFutbol()
        self.recursos_3d = RecursosVisuales3D()
        self.ultimo_golpeador: dict | None = None
        self._poseedor_control_anterior: dict | None = None
        self.ia: IAFutbol | None = None
        self.jugadores: list[dict] = []
        self.poseedor: dict | None = None
        self.arena = Entity(parent=self, enabled=True)
        self.menu_widgets: list[Entity] = []
        self.hud_widgets: list[Entity] = []
        self.hud_marcador: Text | None = None
        self.hud_reloj: Text | None = None
        self.hud_estado: Text | None = None
        self.hud_nombres: dict[str, Text] = {}
        self.radar_puntos: list[Entity] = []
        self.radar_balon: Entity | None = None
        self.overlay_widgets: list[Entity] = []
        self.overlay_title: Text | None = None
        self.overlay_subtitle: Text | None = None
        self.overlay_resume: Button | None = None
        self.overlay_restart: Button | None = None
        self.overlay_menu: Button | None = None
        self.balon: Entity
        self.vista_cenital = False
        self.boton_vista: Button | None = None
        self.panel_marcador: Text | None = None
        self.mensaje_partido = ""
        self.mensaje_tiempo = 0.0
        self.tiempo = 0.0
        self.reloj_partido = 150.0
        self.temporizador_ia = 0.0
        self.pausa = False
        self.fin = False
        # El estadio sirve también como fondo visual del menú principal.
        self._crear_estadio()
        self._crear_balon()
        self.balon.enabled = False
        self._menu()

    # ------------------------------- Menús ---------------------------------
    def _menu(self) -> None:
        """Menú de partido con composición original inspirada en retransmisiones."""
        def panel(x: float, y: float, ancho: float, alto: float, tono,
                  z: float = .02, rotacion: float = 0) -> Entity:
            entidad = Entity(parent=camera.ui, model="quad", position=Vec3(x, y, z),
                             scale=Vec3(ancho, alto, 1), rotation_z=rotacion,
                             color=tono, unlit=True)
            self.menu_widgets.append(entidad)
            return entidad

        def texto(contenido: str, x: float, y: float, escala: float, tono,
                  z: float = -.035) -> Text:
            entidad = Text(contenido, parent=camera.ui, position=Vec3(x, y, z),
                           scale=escala, color=tono, unlit=True)
            self.menu_widgets.append(entidad)
            return entidad

        # Menu original de alto contraste inspirado en futbol de consola 2002.
        panel(0, 0, 2, 2, color.rgba32(3, 22, 36, 72), z=.12)
        panel(-.31, 0, .72, 1.04, color.rgba32(8, 43, 79, 244), z=.07)
        panel(.36, 0, .56, 1.04, color.rgba32(6, 36, 67, 244), z=.07)
        panel(-.31, .476, .72, .014, color.rgb32(255, 202, 48), z=.055)
        panel(-.31, .459, .72, .005, color.rgb32(39, 197, 151), z=.054)

        # Identidad visual propia, sin logos ni recursos de marcas existentes.
        panel(-.612, .414, .045, .045, color.rgb32(255, 202, 48), z=.025)
        texto("R", -.624, .397, .86, color.rgb32(8, 43, 79), z=-.045)
        texto("RETRO CUP  /  2002", -.565, .408, .68, AMARILLO)
        texto("TORNEO INTERNACIONAL", -.565, .342, 1.12, BLANCO)
        texto("Futbol de selecciones - modo clasico", -.565, .293, .58,
              color.rgb32(211, 228, 236))
        texto("ELIGE TU MODO DE PARTIDO", -.565, .225, .52,
              color.rgb32(65, 224, 185))

        modos = [
            ("EXHIBICIÓN  ·  1 VS IA", "Un usuario contra el equipo rival controlado por IA."),
            ("DUELO LOCAL  ·  1 VS 1", "Dos personas juegan en el mismo teclado."),
            ("COOPERATIVO  ·  2 VS IA", "Dos usuarios cooperan frente a un equipo de IA."),
            ("PARTIDO  ·  2 VS 2", "Dos usuarios y dos jugadores independientes de IA."),
        ]
        tonos = [AZUL, color.rgb32(111, 210, 164), AMARILLO, ROJO]
        for indice, (titulo, detalle) in enumerate(modos):
            y = .135 - indice * .122
            activo = indice == 0
            base = color.rgba32(20, 83, 133, 250) if activo else color.rgba32(15, 53, 88, 242)
            panel(-.31, y, .645, .102, base, z=.035)
            panel(-.615, y, .014, .082, tonos[indice], z=.005)
            boton = Button(parent=camera.ui, text="", position=Vec3(-.31, y, -.012),
                           scale=Vec3(.645, .102, 1),
                           color=cast(Any, color.rgba32(22, 43, 59, 0)),
                           highlight_color=color.rgba32(49, 77, 96, 150),
                           pressed_color=tonos[indice],
                           on_click=Func(self.iniciar, indice + 1))
            self.menu_widgets.append(boton)
            texto(f"0{indice + 1}", -.585, y + .012, .62, tonos[indice], z=-.05)
            texto(titulo, -.535, y + .014, .57, BLANCO, z=-.05)
            texto(detalle, -.535, y - .025, .405,
                  color.rgb32(158, 180, 194), z=-.05)

        # Selector nacional editable por teclado o botones de flecha.
        texto("PRESENTACION DEL PARTIDO", .105, .414, .58, AMARILLO)
        texto("SELECCIONES / 7 JUGADORES", .105, .363, .70, BLANCO)
        actual_0 = obtener_seleccion(self.pais_indices[0])
        actual_1 = obtener_seleccion(self.pais_indices[1])
        self.pais_escudos.append(panel(.175, .277, .070, .060,
                                      color.rgb32(*actual_0.camiseta), z=.015))
        self.pais_etiquetas.append(texto(actual_0.sigla, .153, .259, .55, BLANCO, z=-.04))
        self.pais_etiquetas.append(texto(actual_0.nombre, .218, .265, .45,
                                         color.rgb32(207, 224, 237)))
        texto("VS", .352, .264, .57, color.rgb32(149, 170, 184))
        self.pais_escudos.append(panel(.490, .277, .070, .060,
                                      color.rgb32(*actual_1.camiseta), z=.015))
        self.pais_etiquetas.append(texto(actual_1.sigla, .468, .259, .55, BLANCO, z=-.04))
        self.pais_etiquetas.append(texto(actual_1.nombre, .532, .265, .45,
                                         color.rgb32(237, 210, 210)))
        for equipo, x, paso in ((0, .12, -1), (0, .31, 1), (1, .435, -1), (1, .645, 1)):
            boton_pais = Button(parent=camera.ui, text="‹" if paso < 0 else "›",
                                position=Vec3(x, .277, -.025),
                                scale=Vec3(.028, .045, 1),
                                color=cast(Any, color.rgb32(23, 46, 63)),
                                highlight_color=color.rgb32(46, 87, 111),
                                text_color=cast(Any, BLANCO), text_size=cast(Any, .7),
                                on_click=Func(self.cambiar_pais, equipo, paso))
            self.menu_widgets.append(boton_pais)
        texto("A/D local · ←/→ visitante · R sortear equipo IA", .105, .203, .43,
              color.rgb32(148, 177, 191))

        # Campo de vista previa. Cada capa tiene profundidad propia para que
        # el césped, las líneas, las áreas y los jugadores siempre se dibujen.
        campo_x, campo_y = .36, .035
        campo_w, campo_h = .43, .235
        panel(campo_x, campo_y, campo_w, campo_h, color.rgb32(236, 241, 233), z=.025)
        panel(campo_x, campo_y, campo_w - .012, campo_h - .012,
              color.rgb32(30, 113, 69), z=.018)
        for franja in range(6):
            tono = color.rgb32(34, 124, 74) if franja % 2 == 0 else color.rgb32(29, 108, 66)
            franja_x = campo_x - campo_w / 2 + (franja + .5) * (campo_w / 6)
            panel(franja_x, campo_y, campo_w / 6 - .002, campo_h - .014, tono, z=.012)

        # Trazos reglamentarios de la mini cancha.
        borde = color.rgb32(224, 236, 222)
        linea = .0025
        panel(campo_x, campo_y + campo_h / 2 - .009, campo_w - .018, linea, borde, z=.003)
        panel(campo_x, campo_y - campo_h / 2 + .009, campo_w - .018, linea, borde, z=.003)
        panel(campo_x - campo_w / 2 + .009, campo_y, linea, campo_h - .018, borde, z=.003)
        panel(campo_x + campo_w / 2 - .009, campo_y, linea, campo_h - .018, borde, z=.003)
        panel(campo_x, campo_y, linea, campo_h - .02, borde, z=.001)
        for lado in (-1, 1):
            area_x = campo_x + lado * .151
            panel(area_x, campo_y, .002, .112, borde, z=.001)
            panel(area_x - lado * .033, campo_y + .055, .068, .002, borde, z=.001)
            panel(area_x - lado * .033, campo_y - .055, .068, .002, borde, z=.001)
        # Círculo central con segmentos; evita depender de texturas externas.
        for paso in range(20):
            angulo = paso * math.tau / 20
            panel(campo_x + math.cos(angulo) * .038,
                  campo_y + math.sin(angulo) * .038,
                  .013, .002, borde, z=-.002,
                  rotacion=math.degrees(angulo))
        panel(campo_x, campo_y, .009, .009, AMARILLO, z=-.006, rotacion=45)
        # Marcadores romboidales de ambos equipos sobre el esquema táctico.
        posiciones_azules = [
            (-.176, 0), (-.132, -.072), (-.132, .072),
            (-.078, -.070), (-.078, .070), (-.026, -.037), (-.026, .037),
        ]
        posiciones_rojas = [
            (.176, 0), (.132, -.072), (.132, .072),
            (.078, -.070), (.078, .070), (.026, -.037), (.026, .037),
        ]
        for equipo_color, posiciones in ((AZUL, posiciones_azules), (ROJO, posiciones_rojas)):
            for dx, dy in posiciones:
                panel(campo_x + dx, campo_y + dy, .012, .012,
                      equipo_color, z=-.009, rotacion=45)

        # Textura de cancha prerenderizada de assets_3d. Se ajusta al recuadro
        # sin deformarla; el esquema vectorial anterior queda como fallback.
        textura_cancha = self.recursos_3d.textura("cancha")
        if textura_cancha is not None:
            ancho_textura = textura_cancha.width_getter()
            alto_textura = textura_cancha.height_getter()
            if ancho_textura and alto_textura:
                factor = min((campo_w - .012) / ancho_textura,
                             (campo_h - .012) / alto_textura)
                imagen_cancha = Entity(
                    parent=camera.ui, model="quad",
                    texture=cast(Any, textura_cancha),
                    position=Vec3(campo_x, campo_y, -.018),
                    scale=Vec3(ancho_textura * factor, alto_textura * factor, 1),
                    color=BLANCO, unlit=True, double_sided=True,
                )
                self.menu_widgets.append(imagen_cancha)

        self.boton_vista = Button(parent=camera.ui, text="VISTA TÁCTICA  /  CAMBIAR CÁMARA",
                                  position=Vec3(.36, -.145, -.02),
                                  scale=Vec3(.43, .052, 1),
                                  color=cast(Any, color.rgb32(21, 48, 68)),
                                  highlight_color=color.rgb32(38, 79, 103),
                                  pressed_color=AMARILLO,
                                  text_color=cast(Any, BLANCO),
                                  text_size=cast(Any, .62),
                                  on_click=Func(self._alternar_vista))
        self.menu_widgets.append(self.boton_vista)
        self._actualizar_boton_vista()
        texto("V cambia la cámara durante el partido", .105, -.198, .43,
              color.rgb32(143, 168, 183))
        panel(.36, -.275, .50, .002, color.rgba32(147, 171, 185, 100), z=.01)
        texto("CONTROLES LOCALES", .105, -.305, .47, AMARILLO)
        texto("J1 WASD · tiro ESPACIO · pase Q · centro E · robo F · cambiar TAB",
              .105, -.345, .405, color.rgb32(208, 220, 224))
        texto("J2 flechas · tiro CTRL DER. · pase ENTER · centro U · robo ALT · cambiar BKSP",
              .105, -.382, .365, color.rgb32(176, 196, 205))
        texto("1–4  ELEGIR MODO     ·     ESC  SALIR", -.565, -.441, .47,
              color.rgb32(133, 158, 174))

    def _actualizar_boton_vista(self) -> None:
        if self.boton_vista is not None:
            self.boton_vista.text = ("CÁMARA  ·  TRANSMISIÓN 3D" if self.vista_cenital
                                     else "CÁMARA  ·  VISTA TÁCTICA")

    def cambiar_pais(self, equipo: int, paso: int) -> None:
        """Actualiza la selección antes de iniciar el partido."""
        self.pais_indices[equipo] = (self.pais_indices[equipo] + paso) % len(SELECCIONES)
        seleccion = obtener_seleccion(self.pais_indices[equipo])
        self.pais_etiquetas[equipo * 2].text = seleccion.sigla
        self.pais_etiquetas[equipo * 2 + 1].text = seleccion.nombre
        self.pais_escudos[equipo].color = color.rgb32(*seleccion.camiseta)

    def _configurar_camara(self) -> None:
        """Alterna entre retransmisión inclinada y táctica cenital tipo 2D."""
        if self.vista_cenital:
            camera.position = Vec3(0, 31, 0)
            camera.rotation = Vec3(90, 0, 0)
            camera.fov = 42
        else:
            camera.position = Vec3(0, 22, -23.5)
            camera.rotation = Vec3(43, 0, 0)
            camera.fov = 50
        for jugador in self.jugadores:
            cuerpo = jugador.get("cuerpo")
            sombra = jugador.get("sombra")
            if cuerpo is not None:
                cuerpo.enabled = True
            if sombra is not None:
                sombra.enabled = True

    def _alternar_vista(self) -> None:
        self.vista_cenital = not self.vista_cenital
        self._configurar_camara()
        self._actualizar_boton_vista()

    def _crear_hud(self) -> None:
        """Interfaz de retransmisión y radar táctico con acabado deportivo."""
        widgets: list[Entity] = []

        def panel(
            x: float, y: float, ancho: float, alto: float, tono: Any,
            z: float = .025, rotacion: float = 0.0,
        ) -> Entity:
            e = Entity(parent=camera.ui, model="quad", position=Vec3(x, y, z),
                       scale=Vec3(ancho, alto, 1), rotation_z=rotacion,
                       color=tono, unlit=True)
            widgets.append(e)
            return e

        def texto(contenido, x, y, escala, tono=BLANCO, z=-.035):
            e = Text(contenido, parent=camera.ui, position=Vec3(x, y, z),
                     scale=escala, color=tono, unlit=True)
            widgets.append(e)
            return e

        def linea(
            x1: float, y1: float, x2: float, y2: float,
            grosor: float, tono: Any, z: float = .006,
        ) -> Entity:
            dx, dy = x2 - x1, y2 - y1
            return panel((x1 + x2) / 2, (y1 + y2) / 2,
                         math.hypot(dx, dy), grosor, tono, z,
                         math.degrees(math.atan2(dy, dx)))

        # Marcador de televisión con placas de equipo y reloj integrado.
        panel(-.33, .436, .68, .092, color.rgba32(8, 18, 29, 242))
        panel(-.33, .390, .68, .004, color.rgb32(217, 177, 75), z=.009)
        panel(-.33, .479, .68, .003, color.rgb32(49, 139, 149), z=.009)
        texto("TORNEO INTERNACIONAL  ·  EN VIVO", -.33, .463, .25,
              color.rgb32(178, 203, 211))
        self.hud_marcador = texto("0  —  0", -.33, .407, .72, BLANCO)

        for equipo, x_bandera, x_sigla in ((0, -.59, -.505), (1, -.07, -.155)):
            tono_marco = color.rgb32(227, 232, 226)
            panel(x_bandera, .431, .047, .039, tono_marco, z=.007)
            bandera = obtener_seleccion(self.pais_indices[equipo]).bandera
            for franja, tono in enumerate(bandera):
                e = Entity(
                    parent=camera.ui, model="quad",
                    position=Vec3(x_bandera, .431 + (.013 - franja * .013), -.012),
                    scale=Vec3(.043, .0125, 1), color=color.rgb32(*tono), unlit=True,
                )
                widgets.append(e)
            sigla = obtener_seleccion(self.pais_indices[equipo]).sigla
            texto(sigla, x_sigla, .423, .27, color.rgb32(220, 229, 232))

        panel(.50, .436, .28, .092, color.rgba32(8, 18, 29, 242))
        panel(.50, .390, .28, .004, color.rgb32(217, 177, 75), z=.009)
        panel(.50, .479, .28, .003, color.rgb32(49, 139, 149), z=.009)
        texto("TIEMPO DE JUEGO", .50, .463, .24,
              color.rgb32(178, 203, 211))
        self.hud_reloj = texto("02:30", .50, .407, .62, BLANCO)
        self.hud_estado = texto("", 0, .345, .42,
                                color.rgb32(240, 219, 151))

        # Etiquetas discretas para los controles locales.
        for control, x in (("H1", -.455), ("H2", .455)):
            panel(x, -.418, .365, .052, color.rgba32(8, 22, 31, 226))
            panel(x, -.444, .365, .002, color.rgb32(49, 139, 149), z=.008)
            e = texto("", x, -.424, .40, BLANCO)
            self.hud_nombres[control] = e

        # Tarjeta del radar: hierba a franjas, marcas reglamentarias y
        # puntos redondos para leer rápido la forma de ambos equipos.
        radar_x, radar_y, radar_w, radar_h = 0.0, -.365, .265, .145
        panel(radar_x, -.345, .36, .222,
              color.rgba32(7, 19, 27, 241), z=.03)
        panel(radar_x, -.454, .36, .003,
              color.rgb32(217, 177, 75), z=.008)
        texto("RADAR TÁCTICO", radar_x, -.244, .25,
              color.rgb32(202, 219, 223))
        panel(radar_x, radar_y, radar_w, radar_h,
              color.rgb32(26, 104, 65), z=.019)
        for franja in range(8):
            alto_franja = radar_h / 8
            tono_franja = (color.rgb32(35, 119, 73) if franja % 2 == 0
                           else color.rgb32(27, 106, 67))
            panel(radar_x, radar_y - radar_h / 2 + alto_franja * (franja + .5),
                  radar_w, alto_franja, tono_franja, z=.014)

        lineas = color.rgb32(227, 236, 219)
        x0, x1 = radar_x - radar_w / 2, radar_x + radar_w / 2
        y0, y1 = radar_y - radar_h / 2, radar_y + radar_h / 2
        linea(x0, y0, x1, y0, .0022, lineas)
        linea(x1, y0, x1, y1, .0022, lineas)
        linea(x1, y1, x0, y1, .0022, lineas)
        linea(x0, y1, x0, y0, .0022, lineas)
        linea(radar_x, y0, radar_x, y1, .0018, lineas)
        for segmento_circulo in range(32):
            a0 = 2 * math.pi * segmento_circulo / 32
            a1 = 2 * math.pi * (segmento_circulo + 1) / 32
            radio_x, radio_y = radar_w * .085, radar_h * .16
            linea(radar_x + math.cos(a0) * radio_x,
                  radar_y + math.sin(a0) * radio_y,
                  radar_x + math.cos(a1) * radio_x,
                  radar_y + math.sin(a1) * radio_y,
                  .0018, lineas)
        for lado in (-1, 1):
            caja_x = radar_x + lado * radar_w * .405
            linea(caja_x, radar_y - radar_h * .30,
                  caja_x, radar_y + radar_h * .30, .002, lineas)
            linea(radar_x + lado * radar_w * .24, radar_y - radar_h * .30,
                  caja_x, radar_y - radar_h * .30, .002, lineas)
            linea(radar_x + lado * radar_w * .24, radar_y + radar_h * .30,
                  caja_x, radar_y + radar_h * .30, .002, lineas)
            area_x = radar_x + lado * radar_w * .465
            linea(area_x, radar_y - radar_h * .17,
                  area_x, radar_y + radar_h * .17, .002, lineas)
            linea(radar_x + lado * radar_w * .37, radar_y - radar_h * .17,
                  area_x, radar_y - radar_h * .17, .002, lineas)
            linea(radar_x + lado * radar_w * .37, radar_y + radar_h * .17,
                  area_x, radar_y + radar_h * .17, .002, lineas)

        self.radar_puntos = []
        for jugador in self.jugadores:
            tono = (color.rgb32(71, 181, 255) if jugador["equipo"] == 0
                    else color.rgb32(255, 91, 92))
            punto = Entity(parent=camera.ui, model="sphere",
                           position=Vec3(0, 0, -.026), scale=Vec3(.013, .013, .009),
                           color=color.rgb32(245, 239, 215), unlit=True)
            Entity(parent=punto, model="sphere", position=Vec3(0, 0, -.10),
                   scale=Vec3(.66, .66, .58), color=tono, unlit=True)
            widgets.append(punto)
            self.radar_puntos.append(punto)
        self.radar_balon = Entity(parent=camera.ui, model="sphere",
                                  position=Vec3(0, 0, -.032),
                                  scale=Vec3(.015, .015, .010),
                                  color=color.rgb32(255, 195, 66), unlit=True)
        Entity(parent=self.radar_balon, model="sphere",
               position=Vec3(0, 0, -.10), scale=Vec3(.55, .55, .55),
               color=BLANCO, unlit=True)
        widgets.append(self.radar_balon)
        panel(0, -.486, .80, .018, color.rgba32(7, 17, 25, 215), z=.025)
        texto("WASD / FLECHAS  ·  PASE  ·  TIRO  ·  V: CÁMARA",
              0, -.488, .26, color.rgb32(207, 220, 223), z=-.035)
        self.hud_widgets = widgets
        self._crear_overlay_partido()


    def _crear_overlay_partido(self) -> None:
        """Crea un menú de pausa/final con botones para seguir, reiniciar o salir al menú."""
        sombra = Entity(parent=camera.ui, model="quad", position=Vec3(0, 0, -.20),
                        scale=Vec3(2, 2, 1), color=color.rgba32(2, 7, 13, 205), unlit=True)
        tarjeta = Entity(parent=camera.ui, model="quad", position=Vec3(0, 0, -.23),
                         scale=Vec3(.72, .65, 1), color=color.rgba32(10, 25, 40, 250), unlit=True)
        acento = Entity(parent=camera.ui, model="quad", position=Vec3(0, .304, -.25),
                        scale=Vec3(.72, .008, 1), color=AZUL, unlit=True)
        self.overlay_title = Text("PAUSA", parent=camera.ui, y=.19, z=-.28,
                                  scale=1.65, color=AMARILLO, unlit=True)
        self.overlay_subtitle = Text("El partido está detenido", parent=camera.ui,
                                     y=.105, z=-.28, scale=.72,
                                     color=color.rgb32(190, 207, 216), unlit=True)
        self.overlay_resume = Button(parent=camera.ui, text="CONTINUAR JUGANDO",
                                     position=Vec3(0, .025, -.31),
                                     scale=Vec3(.48, .075, 1),
                                     color=cast(Any, color.rgb32(25, 78, 101)),
                                     highlight_color=color.rgb32(39, 115, 143),
                                     pressed_color=AZUL, text_color=cast(Any, BLANCO),
                                     text_size=cast(Any, .8),
                                     on_click=Func(self._reanudar_partido))
        self.overlay_restart = Button(parent=camera.ui, text="REINICIAR PARTIDO",
                                      position=Vec3(0, -.075, -.31),
                                      scale=Vec3(.48, .075, 1),
                                      color=cast(Any, color.rgb32(24, 47, 64)),
                                      highlight_color=color.rgb32(45, 77, 94),
                                      pressed_color=AMARILLO, text_color=cast(Any, BLANCO),
                                      text_size=cast(Any, .8),
                                      on_click=Func(self._reiniciar_partido))
        self.overlay_menu = Button(parent=camera.ui, text="VOLVER A SELECCIÓN",
                                   position=Vec3(0, -.175, -.31),
                                   scale=Vec3(.48, .075, 1),
                                   color=cast(Any, color.rgb32(24, 47, 64)),
                                   highlight_color=color.rgb32(45, 77, 94),
                                   pressed_color=ROJO, text_color=cast(Any, BLANCO),
                                   text_size=cast(Any, .8),
                                   on_click=Func(self._volver_al_menu))
        self.overlay_widgets = [sombra, tarjeta, acento, self.overlay_title,
                                self.overlay_subtitle, self.overlay_resume,
                                self.overlay_restart, self.overlay_menu]
        for widget in self.overlay_widgets:
            widget.enabled = False

    def _actualizar_overlay_partido(self) -> None:
        visible = self.estado == "partido" and (self.pausa or self.fin)
        for widget in self.overlay_widgets:
            widget.enabled = visible
        if (not visible or self.overlay_title is None
                or self.overlay_subtitle is None):
            return
        if self.fin:
            self.overlay_title.text = "PARTIDO FINALIZADO"
            self.overlay_subtitle.text = "Puedes jugar la revancha o volver al menú de selecciones."
            if self.overlay_resume is not None:
                self.overlay_resume.enabled = False
            if self.overlay_restart is not None:
                self.overlay_restart.text = "JUGAR DE NUEVO"
        else:
            self.overlay_title.text = "PAUSA"
            self.overlay_subtitle.text = "Partido detenido · P continúa · R reinicia · Esc menú"
            if self.overlay_resume is not None:
                self.overlay_resume.enabled = True
            if self.overlay_restart is not None:
                self.overlay_restart.text = "REINICIAR PARTIDO"

    def _reanudar_partido(self) -> None:
        if self.estado == "partido" and not self.fin:
            self.pausa = False
            self._actualizar_overlay_partido()

    def _reiniciar_partido(self) -> None:
        if self.estado == "partido":
            self.iniciar(self.modo)

    def _volver_al_menu(self) -> None:
        if self.ia and not self.ia.log.done:
            self.ia.log.finish()
        self.estado = "menu"
        self.pausa = self.fin = False
        for jugador in self.jugadores:
            jugador["e"].enabled = False
        self.balon.enabled = False
        self._hud_visible(False)
        self._actualizar_overlay_partido()
        for widget in self.menu_widgets:
            widget.enabled = True

    def _hud_visible(self, visible: bool) -> None:
        for widget in self.hud_widgets:
            widget.enabled = visible

    # ---------------------------- Estadio/campo -----------------------------
    def _crear_estadio(self) -> None:
        # El shader de sombras se aplica a la geometría 3D que se crea a
        # continuación; se restaura al terminar para que el menú 2D no cambie.
        entidad_ursina = cast(Any, Entity)
        shader_anterior = entidad_ursina.default_shader
        entidad_ursina.default_shader = cast(Any, lit_with_shadows_shader)

        # Iluminación y suelo del estadio.
        # Césped perimetral a franjas para que el estadio tenga profundidad
        # futbolera incluso fuera de las líneas de juego.
        Entity(parent=self.arena, model="plane", scale=Vec3(46, 1, 34),
               color=color.rgb32(15, 45, 43), position=Vec3(0, -.32, 0), unlit=True)
        for franja in range(12):
            ancho_franja = 46 / 12
            x_franja = -23 + ancho_franja * (franja + .5)
            tono_franja = (color.rgb32(29, 70, 53) if franja % 2 == 0
                           else color.rgb32(23, 60, 48))
            Entity(parent=self.arena, model="plane",
                   scale=Vec3(ancho_franja, 1, 34), color=tono_franja,
                   position=Vec3(x_franja, -.315, 0), unlit=True)
        textura_cesped = self.recursos_3d.textura("cesped")
        Entity(parent=self.arena, model="plane", scale=Vec3(LARGO, 1, ANCHO),
               texture=cast(Any, textura_cesped),
               color=BLANCO if textura_cesped is not None else VERDE_1,
               position=Vec3(0, -.015, 0))
        if textura_cesped is None:
            for i in range(12):
                x = -LARGO / 2 + i * LARGO / 12 + LARGO / 24
                Entity(parent=self.arena, model="cube",
                       color=VERDE_1 if i % 2 == 0 else VERDE_2,
                       position=Vec3(x, -.005, 0),
                       scale=Vec3(LARGO / 12, .02, ANCHO - .12))

        # Líneas blancas reglamentarias.
        x0, x1, z0, z1 = -LARGO / 2, LARGO / 2, -ANCHO / 2, ANCHO / 2
        segmento((x0, z0), (x1, z0), parent=self.arena)
        segmento((x0, z1), (x1, z1), parent=self.arena)
        segmento((x0, z0), (x0, z1), parent=self.arena)
        segmento((x1, z0), (x1, z1), parent=self.arena)
        segmento((0, z0), (0, z1), parent=self.arena)
        # Círculo central y punto de saque dibujados como líneas segmentadas.
        self._circulo((0, .012, 0), 2.0, 48, parent=self.arena)
        Entity(parent=self.arena, model="sphere", color=BLANCO,
               scale=Vec3(.12, .12, .12),
               position=Vec3(0, .02, 0))
        for lado in (-1, 1):
            x = lado * 8.0
            xa = lado * 12.0
            # Área grande (16 x 6), área chica (5 x 3.2), penal y semicírculo.
            segmento((xa, -3), (x, -3), parent=self.arena)
            segmento((x, -3), (x, 3), parent=self.arena)
            segmento((x, 3), (xa, 3), parent=self.arena)
            xc = lado * 10.35
            segmento((xa, -1.65), (xc, -1.65), parent=self.arena)
            segmento((xc, -1.65), (xc, 1.65), parent=self.arena)
            segmento((xc, 1.65), (xa, 1.65), parent=self.arena)
            Entity(parent=self.arena, model="sphere", color=BLANCO,
                   scale=Vec3(.11, .11, .11),
                   position=Vec3(lado * 8.1, .02, 0))
            self._arco_penal(lado, parent=self.arena)
            self._portería(lado, parent=self.arena)

        self._arcos_esquina(parent=self.arena)
        if getattr(self, "cancha_de_barrio", False):
            self._crear_entorno_barrio()
            sol = DirectionalLight(parent=self.arena, y=14, z=-7,
                                   shadows=True,
                                   shadow_map_resolution=Vec2(1024, 1024))
            sol.look_at(Vec3(0, 0, 0))
            sol.update_bounds(self.arena)
            AmbientLight(parent=self.arena,
                         color=color.rgba32(172, 179, 168, 255))
            entidad_ursina.default_shader = shader_anterior
            self._configurar_camara()
            return

        # Túneles de acceso, pasillos y barandillas dan escala a las cuatro gradas.
        for lado in (-1, 1):
            for x in (-7.15, 0, 7.15):
                Entity(parent=self.arena, model="cube",
                       color=color.rgb32(12, 25, 35),
                       position=Vec3(x, .70, lado * 7.91),
                       scale=Vec3(1.34, 1.24, .24), unlit=True)
                Entity(parent=self.arena, model="cube",
                       color=color.rgb32(101, 133, 148),
                       position=Vec3(x, 1.36, lado * 7.91),
                       scale=Vec3(1.48, .10, .31), unlit=True)
                Entity(parent=self.arena, model="cube",
                       color=color.rgb32(31, 86, 120),
                       position=Vec3(x, .26, lado * 7.91),
                       scale=Vec3(1.12, .12, .28), unlit=True)

        # Cubiertas, vigas y soportes enmarcan las gradas sin invadir la vista
        # de la banda cercana ni ocultar la línea de juego.
        for lado in (-1, 1):
            Entity(parent=self.arena, model="cube",
                   color=color.rgb32(24, 40, 55),
                   position=Vec3(0, 6.02, lado * 13.50),
                   scale=Vec3(30.2, .30, 2.45), unlit=True)
            Entity(parent=self.arena, model="cube",
                   color=color.rgb32(49, 70, 83),
                   position=Vec3(0, 5.77, lado * 12.30),
                   scale=Vec3(29.6, .12, .14), unlit=True)
            Entity(parent=self.arena, model="cube",
                   color=color.rgb32(21, 37, 50),
                   position=Vec3(0, 5.77, lado * 14.66),
                   scale=Vec3(30.0, .12, .14), unlit=True)
            for x in (-13.8, -8.3, -2.8, 2.8, 8.3, 13.8):
                Entity(parent=self.arena, model=Cylinder(),
                       color=color.rgb32(57, 73, 83),
                       position=Vec3(x, 3.08, lado * 13.65),
                       scale=Vec3(.14, 5.75, .14), unlit=True)
                Entity(parent=self.arena, model="cube",
                       color=color.rgb32(73, 91, 103),
                       position=Vec3(x, 5.90, lado * 13.02),
                       scale=Vec3(.14, .14, 1.30), unlit=True)

            # Franja de luz bajo el alero y soportes del anillo superior.
            Entity(parent=self.arena, model="cube",
                   color=color.rgb32(165, 199, 212),
                   position=Vec3(0, 5.58, lado * 12.25),
                   scale=Vec3(27.8, .055, .10), unlit=True)

        for lado in (-1, 1):
            Entity(parent=self.arena, model="cube",
                   color=color.rgb32(24, 40, 55),
                   position=Vec3(lado * 19.90, 6.02, 0),
                   scale=Vec3(2.35, .30, 16.8), unlit=True)
            for z in (-6.2, 0, 6.2):
                Entity(parent=self.arena, model=Cylinder(),
                       color=color.rgb32(57, 73, 83),
                       position=Vec3(lado * 19.35, 3.08, z),
                       scale=Vec3(.14, 5.75, .14), unlit=True)

        # Pantalla de estadio elevada sobre la grada opuesta. El marcador del
        # partido sigue siendo el HUD independiente de la cámara.
        textura_estadio = self.recursos_3d.textura("estadio")
        Entity(parent=self.arena, model="cube", color=color.rgb32(7, 17, 27),
               position=Vec3(0, 5.18, 12.78), scale=Vec3(6.2, 2.12, .32),
               unlit=True)
        if textura_estadio is not None:
            Entity(parent=self.arena, model="quad", texture=cast(Any, textura_estadio),
                   position=Vec3(0, 5.36, 12.59), rotation=Vec3(0, 180, 0),
                   scale=Vec3(5.72, 1.57, 1), color=BLANCO, unlit=True,
                   double_sided=True)
            Entity(parent=self.arena, model="quad",
                   position=Vec3(0, 4.42, 12.57), rotation=Vec3(0, 180, 0),
                   scale=Vec3(5.72, .30, 1), color=color.rgba32(4, 13, 24, 230),
                   unlit=True, double_sided=True)
        self.panel_marcador = Text("RETRO CUP / 2002", parent=self.arena,
                                   position=Vec3(0, 4.34, 12.55),
                                   rotation=Vec3(0, 180, 0), scale=1.05,
                                   color=AMARILLO, unlit=True)
        # Publicidad LED geométrica alrededor de todo el terreno de juego.
        tonos_panel = [
            color.rgb32(27, 109, 161), color.rgb32(25, 126, 91),
            color.rgb32(181, 115, 43), color.rgb32(163, 48, 61),
        ]
        for i, x in enumerate(range(-11, 12, 2)):
            for lado in (-1, 1):
                Entity(parent=self.arena, model="cube", color=tonos_panel[i % len(tonos_panel)],
                       position=Vec3(x, .22, lado * 7.22),
                       scale=Vec3(1.92, .42, .12), unlit=True)
                Entity(parent=self.arena, model="cube",
                       color=color.rgb32(194, 217, 219),
                       position=Vec3(x, .45, lado * 7.22),
                       scale=Vec3(1.92, .035, .13), unlit=True)
        for lado in (-1, 1):
            for z in (-5.7, -4.2, 4.2, 5.7):
                Entity(parent=self.arena, model="cube",
                       color=tonos_panel[(int(abs(z) * 2) + lado) % len(tonos_panel)],
                       position=Vec3(lado * 12.10, .22, z),
                       scale=Vec3(.12, .42, 1.35), unlit=True)

        self._multitud()
        # Torres y bancos de reflectores visibles, con luminarias orientadas
        # hacia la cancha.
        for x in (-20.1, 20.1):
            for z in (-14.8, 14.8):
                Entity(parent=self.arena, model=Cylinder(),
                       color=color.rgb32(62, 79, 90),
                       position=Vec3(x, 5.18, z), scale=Vec3(.14, 10.2, .14),
                       unlit=True)
                Entity(parent=self.arena, model="cube",
                       color=color.rgb32(31, 47, 60),
                       position=Vec3(x, 10.18, z),
                       scale=Vec3(1.55, .38, .56), unlit=True)
                for indice in range(7):
                    lx = x - .54 + indice * .18
                    Entity(parent=self.arena, model="cube",
                           color=color.rgb32(224, 239, 239),
                           position=Vec3(lx, 10.18, z - math.copysign(.23, z)),
                           scale=Vec3(.13, .17, .045), unlit=True)
                Entity(parent=self.arena, model="cube",
                       color=color.rgb32(94, 116, 125),
                       position=Vec3(x, 9.91, z),
                       scale=Vec3(1.78, .08, .68), unlit=True)
        # Banderines de esquina completan la lectura de estadio.
        for x in (-11.7, 11.7):
            for z in (-6.7, 6.7):
                Entity(parent=self.arena, model=Cylinder(), color=color.rgb32(232, 236, 222),
                       position=Vec3(x, .36, z), scale=Vec3(.035, .72, .035))
                Entity(parent=self.arena, model="cube",
                       color=AZUL if (x * z) > 0 else ROJO,
                       position=Vec3(x + .13, .60, z), scale=Vec3(.27, .16, .025))
        sol = DirectionalLight(parent=self.arena, y=12, z=-5,
                               shadows=True, shadow_map_resolution=Vec2(512, 512))
        sol.look_at(Vec3(0, 0, 0))
        sol.update_bounds(self.arena)
        AmbientLight(parent=self.arena, color=color.rgba32(178, 190, 180, 255))
        entidad_ursina.default_shader = shader_anterior
        self._configurar_camara()

    def _crear_entorno_barrio(self) -> None:
        """Cancha comunitaria cercada entre casas, muros y luminarias sencillas."""
        metal = color.rgb32(72, 87, 78)
        alambre = color.rgb32(105, 119, 106)
        concreto = color.rgb32(126, 124, 111)
        concreto_claro = color.rgb32(164, 158, 139)

        # Muros bajos y malla alrededor del campo, sin graderias ni cubiertas.
        for lado in (-1, 1):
            z = lado * 8.10
            techo_malla = 1.78 if lado == -1 else 3.08
            centro_malla = (.84 + techo_malla) / 2
            alto_malla = techo_malla - .84
            cables_y = (
                (.97, 1.23, 1.50, 1.78) if lado == -1
                else (.97, 1.48, 2.02, 2.56, 3.08)
            )
            Entity(parent=self.arena, model="cube", color=concreto,
                   position=Vec3(0, .42, z), scale=Vec3(27.2, .84, .30))
            Entity(parent=self.arena, model="cube", color=concreto_claro,
                   position=Vec3(0, .86, z), scale=Vec3(27.2, .08, .34))
            for x in range(-13, 14, 2):
                if lado == -1 and abs(x) <= 1:
                    continue
                Entity(parent=self.arena, model=Cylinder(resolution=8),
                       color=metal, position=Vec3(x, centro_malla, z),
                       scale=Vec3(.07, alto_malla, .07), unlit=True)
            for y in cables_y:
                Entity(parent=self.arena, model="cube", color=alambre,
                       position=Vec3(0, y, z), scale=Vec3(27.2, .018, .018),
                       unlit=True)
            for x in [i * .55 for i in range(-24, 25)]:
                if lado == -1 and abs(x) < 1.45:
                    continue
                Entity(parent=self.arena, model="cube", color=alambre,
                       position=Vec3(x, centro_malla, z),
                       scale=Vec3(.014, alto_malla, .014),
                       unlit=True)

        for lado in (-1, 1):
            x = lado * 13.72
            Entity(parent=self.arena, model="cube", color=concreto,
                   position=Vec3(x, .42, 0), scale=Vec3(.32, .84, 16.2))
            for z in range(-8, 9, 2):
                Entity(parent=self.arena, model=Cylinder(resolution=8),
                       color=metal, position=Vec3(x, 2.02, z),
                       scale=Vec3(.07, 2.35, .07), unlit=True)
            for y in (.97, 1.48, 2.02, 2.56, 3.08):
                Entity(parent=self.arena, model="cube", color=alambre,
                       position=Vec3(x, y, 0), scale=Vec3(.018, .018, 16.2),
                       unlit=True)
            for z in [i * .55 for i in range(-14, 15)]:
                Entity(parent=self.arena, model="cube", color=alambre,
                       position=Vec3(x, 2.02, z), scale=Vec3(.014, 2.30, .014),
                       unlit=True)

        # Tres fachadas de barrio con ventanas, azoteas y depositos de agua.
        fachadas = (
            (-9.0, 8.2, 5.4, color.rgb32(158, 116, 91)),
            (0.0, 8.0, 6.5, color.rgb32(179, 167, 137)),
            (9.0, 8.2, 5.0, color.rgb32(128, 146, 123)),
        )
        for indice, (x, ancho, alto, tono) in enumerate(fachadas):
            z = 14.25
            Entity(parent=self.arena, model="cube", color=tono,
                   position=Vec3(x, alto / 2, z), scale=Vec3(ancho, alto, 4.5))
            Entity(parent=self.arena, model="cube",
                   color=color.rgb32(77, 76, 67),
                   position=Vec3(x, alto + .08, z), scale=Vec3(ancho + .35, .20, 4.8))
            for columna in range(3):
                wx = x + (columna - 1) * 2.15
                for piso in range(1, max(2, int(alto // 2))):
                    wy = piso * 1.65
                    Entity(parent=self.arena, model="cube",
                           color=color.rgb32(58, 91, 104),
                           position=Vec3(wx, wy, z - 2.27), scale=Vec3(1.05, .92, .06))
                    Entity(parent=self.arena, model="cube",
                           color=color.rgb32(202, 193, 164),
                           position=Vec3(wx, wy, z - 2.31), scale=Vec3(.06, .98, .035))
            if indice != 1:
                Entity(parent=self.arena, model=Cylinder(resolution=10),
                       color=color.rgb32(87, 115, 92),
                       position=Vec3(x + ancho * .28, alto + .62, z),
                       scale=Vec3(.68, 1.05, .68))

        # Mural pintado y letrero comunitario para dar identidad al potrero.
        Entity(parent=self.arena, model="cube", color=color.rgb32(43, 91, 83),
               position=Vec3(-8.4, 1.72, 7.91), scale=Vec3(4.0, 1.22, .06), unlit=True)
        for indice, tono in enumerate((
            color.rgb32(234, 190, 62), color.rgb32(206, 83, 61),
            color.rgb32(70, 139, 176),
        )):
            Entity(parent=self.arena, model="cube", color=tono,
                   position=Vec3(-9.25 + indice * .82, 1.78, 7.86),
                   scale=Vec3(.55, .74, .025), rotation_z=(indice - 1) * 9,
                   unlit=True)
        Text("CANCHA DEL BARRIO", parent=self.arena,
             position=Vec3(0, 5.35, 11.94), rotation=Vec3(0, 180, 0),
             scale=.82, color=color.rgb32(238, 218, 160), unlit=True)

        # Postes de alumbrado de calle en las esquinas.
        for x in (-14.8, 14.8):
            for z in (-7.8, 7.8):
                Entity(parent=self.arena, model=Cylinder(resolution=8),
                       color=metal, position=Vec3(x, 3.55, z),
                       scale=Vec3(.10, 7.0, .10), unlit=True)
                Entity(parent=self.arena, model="cube",
                       color=color.rgb32(69, 78, 69),
                       position=Vec3(x, 6.95, z - math.copysign(.32, z)),
                       scale=Vec3(.76, .24, .65), unlit=True)
                for foco in range(3):
                    Entity(parent=self.arena, model="cube",
                           color=color.rgb32(243, 232, 184),
                           position=Vec3(x - .23 + foco * .23, 6.90, z - math.copysign(.57, z)),
                           scale=Vec3(.13, .09, .04), unlit=True)

        # Arboles del vecindario enmarcan las fachadas lejanas.
        for x in (-18.8, 18.8):
            Entity(parent=self.arena, model=Cylinder(resolution=7),
                   color=color.rgb32(94, 77, 57),
                   position=Vec3(x, 2.0, 12.3), scale=Vec3(.23, 4.0, .23))
            Entity(parent=self.arena, model="sphere",
                   color=color.rgb32(59, 104, 68),
                   position=Vec3(x, 4.5, 12.3), scale=Vec3(2.2, 2.4, 2.0))

    def _circulo(self, centro: tuple[float, float, float], radio: float,
                 pasos: int, parent) -> None:
        cx, y, cz = centro
        puntos = [(cx + math.cos(i * 2 * math.pi / pasos) * radio,
                   cz + math.sin(i * 2 * math.pi / pasos) * radio) for i in range(pasos + 1)]
        for a, b in zip(puntos, puntos[1:]):
            segmento(a, b, grosor=.04, y=y, parent=parent)

    def _arco_penal(self, lado: int, parent) -> None:
        cx, radio = lado * 8.1, 2.0
        puntos = []
        # Sólo se dibuja el arco que queda fuera del rectángulo del área.
        inicio = math.pi / 2 if lado == 1 else -math.pi / 2
        for i in range(25):
            ang = inicio + i * math.pi / 24
            puntos.append((cx + math.cos(ang) * radio, math.sin(ang) * radio))
        for a, b in zip(puntos, puntos[1:]):
            segmento(a, b, y=.012, parent=parent)

    def _arcos_esquina(self, parent) -> None:
        """Dibuja los cuatro cuartos de circulo de las esquinas del campo."""
        radio = .42
        pasos = 10
        for lado_x in (-1, 1):
            for lado_z in (-1, 1):
                puntos = [
                    (lado_x * (LARGO / 2 - radio * math.cos(angulo)),
                     lado_z * (ANCHO / 2 - radio * math.sin(angulo)))
                    for angulo in (i * math.pi / (2 * pasos) for i in range(pasos + 1))
                ]
                for a, b in zip(puntos, puntos[1:]):
                    segmento(a, b, grosor=.04, y=.012, parent=parent)

    def _portería(self, lado: int, parent) -> None:
        x = lado * 12.35
        fondo_x = lado * 13.55
        centro_x = (x + fondo_x) / 2
        profundidad = abs(fondo_x - x)
        poste = color.rgb32(249, 251, 245)
        tono_base = color.rgb32(93, 111, 105)
        tono_red = color.rgb32(201, 224, 219)

        for z in (-1.7, 1.7):
            Entity(parent=parent, model="cube", color=poste,
                   position=Vec3(x, .95, z), scale=Vec3(.22, 1.9, .22), unlit=True)
            Entity(parent=parent, model="cube", color=tono_base,
                   position=Vec3(x, .11, z), scale=Vec3(.28, .22, .28), unlit=True)
            Entity(parent=parent, model="cube", color=poste,
                   position=Vec3(fondo_x, .95, z), scale=Vec3(.12, 1.9, .12),
                   unlit=True)
            Entity(parent=parent, model="cube", color=poste,
                   position=Vec3(centro_x, 1.9, z),
                   scale=Vec3(profundidad, .12, .12), unlit=True)
        Entity(parent=parent, model="cube", color=poste,
               position=Vec3(x, 1.9, 0), scale=Vec3(.22, .22, 3.58), unlit=True)
        Entity(parent=parent, model="cube", color=poste,
               position=Vec3(fondo_x, 1.9, 0), scale=Vec3(.12, .12, 3.4),
               unlit=True)

        # La malla clara y más gruesa define el volumen de la portería.
        for indice in range(9):
            z = -1.68 + indice * .42
            Entity(parent=parent, model="cube", color=tono_red,
                   position=Vec3(fondo_x, .97, z), scale=Vec3(.025, 1.82, .025),
                   unlit=True)
            Entity(parent=parent, model="cube", color=tono_red,
                   position=Vec3(centro_x, 1.93, z),
                   scale=Vec3(profundidad, .024, .024), unlit=True)
        for nivel in range(1, 6):
            yy = .15 + nivel * .34
            Entity(parent=parent, model="cube", color=tono_red,
                   position=Vec3(fondo_x, yy, 0),
                   scale=Vec3(.025, .024, 3.38), unlit=True)
            Entity(parent=parent, model="cube", color=tono_red,
                   position=Vec3(centro_x, yy, -1.70),
                   scale=Vec3(profundidad, .024, .024), unlit=True)
            Entity(parent=parent, model="cube", color=tono_red,
                   position=Vec3(centro_x, yy, 1.70),
                   scale=Vec3(profundidad, .024, .024), unlit=True)
        for z in (-1.7, 1.7):
            for indice in range(5):
                xx = x + (fondo_x - x) * (indice + 1) / 6
                Entity(parent=parent, model="cube", color=tono_red,
                       position=Vec3(xx, .98, z), scale=Vec3(.025, 1.82, .025),
                       unlit=True)

    def _multitud(self) -> None:
        rng = random.Random(29)
        tonos = [color.rgb32(47, 134, 224), color.rgb32(228, 74, 83),
                 color.rgb32(244, 202, 75), color.rgb32(223, 229, 221),
                 color.rgb32(48, 173, 126)]
        for i in range(180):
            color_fan = rng.choice(tonos)
            lado = rng.choice((-1, 1))
            if i % 2:
                x, z = rng.uniform(-13, 13), lado * rng.uniform(8.5, 10.1)
            else:
                x, z = lado * rng.uniform(13, 15.1), rng.uniform(-7.8, 7.8)
            y = rng.choice((.95, 1.5, 2.05))
            Entity(parent=self.arena, model="sphere", color=color_fan,
                   position=Vec3(x, y, z),
                   scale=Vec3(.11, .11, .11), unlit=True)

        for z, nombre, giro in ((-8.66, "hinchas_1", 0), (8.66, "hinchas_2", 180)):
            textura = self.recursos_3d.textura(nombre)
            if textura is None:
                continue
            ancho_textura = textura.width_getter()
            alto_textura = textura.height_getter()
            if not ancho_textura or not alto_textura:
                continue
            ancho = 12.0
            alto = ancho * alto_textura / ancho_textura
            Entity(parent=self.arena, model="quad", texture=cast(Any, textura),
                   position=Vec3(0, .48 + alto / 2, z), rotation=Vec3(0, giro, 0),
                   scale=Vec3(ancho, alto, 1), color=BLANCO, unlit=True,
                   double_sided=True)

    # --------------------------- Inicio y entidades ------------------------
    def iniciar(self, modo: int) -> None:
        self.modo = modo
        if self.ia is None:
            self.ia = IAFutbol()
        elif not self.ia.log.done:
            self.ia.log.finish()
        paises = tuple(obtener_seleccion(i).nombre for i in self.pais_indices)
        self.ia.log.start(modo, paises)
        for widget in self.menu_widgets:
            widget.enabled = False
        # Conserva el estadio del menú y recrea únicamente los actores del partido.
        for jugador in self.jugadores:
            destroy(jugador["e"])
        if self.balon:
            destroy(self.balon)
        self._crear_jugadores()
        self._crear_balon()
        self._iniciar_saque(0)
        if not self.hud_widgets:
            self._crear_hud()
        self._hud_visible(True)
        self.tiempo = self.temporizador_ia = 0.0
        self.reloj_partido = 150.0
        self.pausa = self.fin = False
        self.estado = "partido"
        self._poseedor_control_anterior = None
        self._configurar_camara()
        self.audio.efecto("silbato")
        self._actualizar_overlay_partido()

    def _iniciar_saque(self, equipo: int) -> None:
        """Da el balón al equipo que recibe el gol y devuelve a todos a su posición."""
        signo = 1 if equipo == 0 else -1
        for jugador in self.jugadores:
            jugador["e"].position = Vec3(jugador["casa"][0], 0, jugador["casa"][1])
            jugador["objetivo"] = jugador["casa"]
            jugador["direccion"] = (float(signo), 0.0)
            jugador["e"].rotation_y = math.degrees(math.atan2(-signo, 0))
            jugador["moviendo"] = False
            jugador["velocidad_ia"] = (0.0, 0.0)
            jugador["recarga_robo"] = 0.0

        sacador = max((p for p in self.jugadores if p["equipo"] == equipo
                       and p["indice"] >= 5), key=lambda p: p["indice"], default=None)
        if sacador is not None:
            sacador["e"].position = Vec3(-signo * .72, 0, 0)
            sacador["objetivo"] = (-signo * .72, 0.0)
            sacador["direccion"] = (float(signo), 0.0)
            sacador["e"].rotation_y = math.degrees(math.atan2(-signo, 0))
            sacador["recarga_robo"] = .8
        self.poseedor = sacador
        self.balon.position = Vec3(0, RADIO_BALON, 0)
        self.vel_balon = Vec3(0, 0, 0)
        self.ultimo_golpeador = None
        self._poseedor_control_anterior = None
        self.mensaje_partido = "SAQUE INICIAL · " + obtener_seleccion(self.pais_indices[equipo]).nombre.upper()
        self.mensaje_tiempo = 2.0

    def _crear_jugadores(self) -> None:
        self.jugadores = []
        for equipo in (0, 1):
          seleccion = obtener_seleccion(self.pais_indices[equipo])
          color_equipo = color.rgb32(*seleccion.camiseta)
          color_detalle = color.rgb32(*seleccion.detalle)

          for indice in range(7):
            color_camiseta = (color.rgb32(235, 168, 48) if equipo == 0 else color.rgb32(139, 101, 219)) if indice == 0 else color_equipo
            control = "IA"
            if self.modo == 1 and equipo == 0 and indice == 6:
                control = "H1"
            elif self.modo == 2 and indice == 6:
                control = "H1" if equipo == 0 else "H2"
            elif self.modo in (3, 4) and equipo == 1 and indice in (5, 6):
                control = "H2" if indice == 5 else "H1"
            x, z = formacion_7(equipo, indice, -10.8, 10.8, 0.0, 5.5)
            actor = Entity(parent=self.arena, position=Vec3(x, 0, z))
            cuerpo = Entity(parent=actor)
            # Uniforme deportivo por capas: botines, medias, pantaloneta,
            # camiseta con franjas, brazos articulados y cabeza con cabello.
            sombra = Entity(parent=actor, model="sphere", color=color.rgba32(8, 18, 15, 135),
                            position=Vec3(0, .055, .03), scale=Vec3(.61, .035, .82), unlit=True)
            marcador = Entity(parent=actor, model=Cylinder(resolution=3),
                              color=color.rgb32(66, 160, 255),
                              position=Vec3(0, 1.98, 0), scale=Vec3(.15, .21, .15),
                              rotation_x=180, unlit=True,
                              enabled=control != "IA")
            piernas, brazos = [], []
            color_pantalon = (
                color.rgb32(27, 71, 151) if equipo == 1
                else color.rgb32(231, 233, 224)
            )
            color_media = color.rgb32(235, 239, 231)
            tonos_piel = [color.rgb32(226, 184, 151), color.rgb32(191, 139, 103),
                          color.rgb32(153, 105, 78), color.rgb32(120, 78, 59)]
            tonos_cabello = [color.rgb32(34, 30, 29), color.rgb32(76, 51, 37),
                             color.rgb32(115, 91, 59)]
            color_piel = tonos_piel[(indice + equipo * 2) % len(tonos_piel)]
            color_cabello = tonos_cabello[(indice * 2 + equipo) % len(tonos_cabello)]

            color_guante = color.rgb32(241, 237, 218) if indice == 0 else color_piel
            for lado in (-1, 1):
                cadera = Entity(parent=cuerpo, position=Vec3(lado * .145, .57, 0))
                piernas.append(cadera)
                Entity(parent=cadera, model=Cylinder(resolution=16), color=color_media,
                       position=Vec3(0, -.18, 0), scale=Vec3(.125, .36, .13))
                Entity(parent=cadera, model=Cylinder(resolution=16),
                       color=color_detalle,
                       position=Vec3(0, -.12, 0), scale=Vec3(.132, .055, .135))
                Entity(parent=cadera, model="cube", color=color.rgb32(30, 34, 39),
                       position=Vec3(0, -.39, -.075), scale=Vec3(.17, .105, .27))
                Entity(parent=cadera, model="cube", color=color.rgb32(221, 224, 214),
                       position=Vec3(0, -.355, -.19), scale=Vec3(.17, .035, .035))

                Entity(parent=cadera, model="sphere", color=color.rgb32(23, 27, 32),
                       position=Vec3(0, -.39, -.13), scale=Vec3(.105, .08, .13))
                brazo = Entity(parent=cuerpo, position=Vec3(lado * .255, 1.12, 0),
                               rotation_z=-lado * 8)
                brazos.append(brazo)

                Entity(parent=cuerpo, model="sphere", color=color_camiseta,
                       position=Vec3(lado * .265, 1.08, 0),
                       scale=Vec3(.16, .18, .16))
                Entity(parent=brazo, model=Cylinder(resolution=16), color=color_camiseta,
                       position=Vec3(0, -.18, 0), scale=Vec3(.12, .39, .13))
                Entity(parent=brazo, model="sphere", color=color_guante,
                       position=Vec3(0, -.40, 0), scale=Vec3(.115, .14, .115))
                Entity(parent=brazo, model=Cylinder(resolution=16),
                       color=color_detalle,
                       position=Vec3(0, -.34, 0), scale=Vec3(.126, .055, .132))
            Entity(parent=cuerpo, model=Cylinder(resolution=18), color=color_pantalon,
                   position=Vec3(0, .58, 0), scale=Vec3(.37, .25, .28))
            Entity(parent=cuerpo, model=Cylinder(resolution=18), color=color_camiseta,
                   position=Vec3(0, .98, 0), scale=Vec3(.43, .53, .30))
            Entity(parent=cuerpo, model=Cylinder(resolution=20),
                   color=color_detalle, position=Vec3(0, 1.245, 0),
                   scale=Vec3(.19, .055, .19))
            # Paneles del uniforme visibles desde la cámara de retransmisión.
            Entity(parent=cuerpo, model="cube", color=color_detalle,
                   position=Vec3(0, .99, -.154), scale=Vec3(.055, .48, .018))
            for lado in (-1, 1):
                Entity(parent=cuerpo, model="cube", color=color_detalle,
                       position=Vec3(lado * .185, .96, -.113), scale=Vec3(.035, .37, .015))
            Entity(parent=cuerpo, model="cube", color=color.rgb32(238, 198, 91),
                   position=Vec3(.17, 1.08, -.23), scale=Vec3(.075, .105, .022))
            Entity(parent=cuerpo, model="sphere", color=color.rgb32(36, 100, 80),
                   position=Vec3(.17, 1.08, -.249), scale=Vec3(.043, .066, .012))
            Entity(parent=cuerpo, model=Cylinder(resolution=16), color=color_piel,
                   position=Vec3(0, 1.30, 0), scale=Vec3(.12, .15, .12))
            Entity(parent=cuerpo, model="sphere", color=color_piel,
                   position=Vec3(0, 1.48, 0), scale=Vec3(.29, .34, .28))
            Entity(parent=cuerpo, model="sphere", color=color_cabello,
                   position=Vec3(0, 1.61, .015), scale=Vec3(.30, .16, .29))
            for lado in (-1, 1):
                Entity(parent=cuerpo, model="sphere", color=color.rgb32(34, 36, 34),
                       position=Vec3(lado * .075, 1.49, -.129), scale=Vec3(.025, .025, .018))
                Entity(parent=cuerpo, model="sphere", color=color_piel,
                       position=Vec3(lado * .148, 1.49, 0), scale=Vec3(.055, .07, .045))
            dorsal = Text(str(indice + 1), parent=cuerpo, position=Vec3(0, 1.02, -.164),
                          scale=1.8, color=BLANCO, unlit=True)
            dorsal.billboard = True
            # Jugadores formados por geometría 3D articulada; no se cargan
            # sprites ni imágenes de jugadores desde assets_3d.
            cuerpo.enabled = True
            sombra.enabled = True
            self.jugadores.append({"equipo": equipo, "indice": indice, "control": control,
                                   "nombre": seleccion.once[indice], "e": actor,
                                   "casa": (x, z), "objetivo": (x, z), "stamina": 1.0,
                                   "direccion": (1.0 if equipo == 0 else -1.0, 0.0),
                                   "recarga_robo": 0.0, "piernas": piernas,
                                   "brazos": brazos, "fase": indice * 1.4,
                                   "moviendo": False, "marcador": marcador,
                                   "cuerpo": cuerpo, "sombra": sombra})

    def _animar_jugadores(self, dt: float) -> None:
        """Anima las extremidades de los modelos usados en la vista cenital."""
        for jugador in self.jugadores:
            if jugador["moviendo"]:
                jugador["fase"] += dt * 15.5
                paso = math.sin(jugador["fase"]) * 25
                jugador["e"].y = .025 + abs(math.sin(jugador["fase"] * 2)) * .025
                for indice, pierna in enumerate(jugador["piernas"]):
                    pierna.rotation_z = paso if indice == 0 else -paso
                for indice, brazo in enumerate(jugador["brazos"]):
                    brazo.rotation_z = (-12 if indice == 0 else 12) - (paso * .45 if indice == 0 else -paso * .45)
            else:
                jugador["e"].y = 0
                for indice, pierna in enumerate(jugador["piernas"]):
                    pierna.rotation_z = 0
                for indice, brazo in enumerate(jugador["brazos"]):
                    brazo.rotation_z = -12 if indice == 0 else 12

    def _crear_balon(self) -> None:
        """Balón clásico con paneles cosidos, generado como malla 3D."""
        self.balon = Entity(parent=self.arena, model="sphere",
                            color=color.rgb32(239, 241, 231),
                            scale=Vec3(.42, .42, .42),
                            position=Vec3(0, RADIO_BALON, 0),
                            shader=cast(Any, lit_with_shadows_shader))

        def unidad(v):
            longitud = math.sqrt(sum(componente * componente for componente in v))
            return tuple(componente / longitud for componente in v)

        def sumar(a, b):
            return tuple(a[i] + b[i] for i in range(3))

        def restar(a, b):
            return tuple(a[i] - b[i] for i in range(3))

        def escalar(v, factor):
            return tuple(componente * factor for componente in v)

        def producto(a, b):
            return sum(a[i] * b[i] for i in range(3))

        def cruz(a, b):
            return (a[1] * b[2] - a[2] * b[1],
                    a[2] * b[0] - a[0] * b[2],
                    a[0] * b[1] - a[1] * b[0])

        phi = (1 + math.sqrt(5)) / 2
        vertices_ico = [
            (-1, phi, 0), (1, phi, 0), (-1, -phi, 0), (1, -phi, 0),
            (0, -1, phi), (0, 1, phi), (0, -1, -phi), (0, 1, -phi),
            (phi, 0, -1), (phi, 0, 1), (-phi, 0, -1), (-phi, 0, 1),
        ]
        vertices_ico = [unidad(v) for v in vertices_ico]
        caras_ico = [
            (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
            (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
            (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
            (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
        ]

        cortes = {}
        vecinos = [set() for _ in vertices_ico]
        for cara in caras_ico:
            for i, a in enumerate(cara):
                b = cara[(i + 1) % 3]
                vecinos[a].add(b)
                vecinos[b].add(a)
                cortes[(a, b)] = unidad(sumar(escalar(vertices_ico[a], 2 / 3),
                                                escalar(vertices_ico[b], 1 / 3)))
                cortes[(b, a)] = unidad(sumar(escalar(vertices_ico[b], 2 / 3),
                                                escalar(vertices_ico[a], 1 / 3)))

        vertices_malla, normales, colores, triangulos = [], [], [], []

        def agregar_panel(poligono, tono):
            centro_direccion = unidad(tuple(sum(p[i] for p in poligono)
                                            for i in range(3)))
            normal = centro_direccion
            if producto(cruz(restar(poligono[1], poligono[0]),
                             restar(poligono[2], poligono[0])), normal) < 0:
                poligono = list(reversed(poligono))
            puntos = [unidad(sumar(escalar(centro_direccion, .035),
                                   escalar(p, .965))) for p in poligono]
            centro = escalar(centro_direccion, .505)
            base = len(vertices_malla)
            vertices_malla.append(centro)
            normales.append(centro_direccion)
            rgba = tuple(float(c) for c in tono)
            colores.append(rgba)
            for punto in puntos:
                vertices_malla.append(escalar(punto, .504))
                normales.append(punto)
                colores.append(rgba)
            for i in range(len(puntos)):
                triangulos.append((base, base + 1 + i,
                                   base + 1 + (i + 1) % len(puntos)))

        oscuro = color.rgb32(20, 25, 31)
        claros = (color.rgb32(248, 248, 239), color.rgb32(230, 233, 229))
        for indice, eje in enumerate(vertices_ico):
            referencia = (0, 1, 0) if abs(eje[1]) < .9 else (1, 0, 0)
            base_u = unidad(cruz(eje, referencia))
            base_v = cruz(eje, base_u)
            ordenados = sorted(
                vecinos[indice],
                key=lambda vecino: math.atan2(producto(cortes[(indice, vecino)], base_v),
                                              producto(cortes[(indice, vecino)], base_u)),
            )
            agregar_panel([cortes[(indice, vecino)] for vecino in ordenados],
                          oscuro)
        for indice, cara in enumerate(caras_ico):
            a, b, c = cara
            hexagono = [cortes[(a, b)], cortes[(b, a)],
                        cortes[(b, c)], cortes[(c, b)],
                        cortes[(c, a)], cortes[(a, c)]]
            agregar_panel(hexagono, claros[indice % len(claros)])

        modelo_paneles = Mesh(vertices=vertices_malla, triangles=triangulos,
                              colors=colores, normals=normales)
        Entity(parent=self.balon, model=modelo_paneles, color=BLANCO,
               shader=cast(Any, lit_with_shadows_shader))
        self.vel_balon = Vec3(0, 0, 0)


    # ------------------------------ Controles ------------------------------
    def _movimiento(self, control: str) -> tuple[float, float, bool]:
        teclas_presionadas = cast(Any, held_keys)
        if control == "H1":
            dx = teclas_presionadas["d"] - teclas_presionadas["a"]
            # En Ursina +Z es hacia delante: W y Flecha arriba deben sumar Z.
            dz = teclas_presionadas["w"] - teclas_presionadas["s"]
            correr = teclas_presionadas["left shift"]
        else:
            dx = teclas_presionadas["right arrow"] - teclas_presionadas["left arrow"]
            dz = teclas_presionadas["up arrow"] - teclas_presionadas["down arrow"]
            correr = teclas_presionadas["right shift"]
        return dx, dz, bool(correr)

    def _accion(self, tecla: str) -> None:
        es_j1 = tecla in ("space", "q", "e")
        p = next((j for j in self.jugadores if j["control"] == ("H1" if es_j1 else "H2")), None)
        if p is None:
            return
        accion = "pase" if tecla in ("q", "enter") else "centro" if tecla in ("e", "u") else "tiro"
        companero = self._mejor_companero(p, solo_ataque=accion == "centro")
        self._patear(p, accion, companero)

    def _mejor_companero(self, jugador: dict, solo_ataque: bool = False) -> dict | None:
        """Busca una opción de pase adelantada y con espacio respecto a los rivales."""
        signo = 1 if jugador["equipo"] == 0 else -1
        candidatos = [q for q in self.jugadores if q["equipo"] == jugador["equipo"]
                      and q is not jugador and q["indice"] != 0]
        if solo_ataque:
            candidatos = [q for q in candidatos if signo * (q["e"].x - jugador["e"].x) > 0]
        rivales = [q for q in self.jugadores if q["equipo"] != jugador["equipo"]]
        def puntuacion(q: dict) -> float:
            avance = signo * (q["e"].x - jugador["e"].x)
            distancia = math.hypot(q["e"].x - jugador["e"].x, q["e"].z - jugador["e"].z)
            espacio = min((math.hypot(q["e"].x - r["e"].x, q["e"].z - r["e"].z)
                           for r in rivales), default=99)
            return avance * .55 - distancia * .35 + min(6, espacio) * .8
        return max(candidatos, key=puntuacion, default=None)

    def _cambiar_jugador(self, control: str, objetivo: dict | None = None) -> None:
        """Pasa el control al siguiente jugador de campo, dejando el anterior a la IA."""
        activo = next((p for p in self.jugadores if p["control"] == control), None)
        if activo is None or objetivo is activo:
            return
        ocupados = {p["indice"] for p in self.jugadores
                    if p["equipo"] == activo["equipo"] and p is not activo
                    and p["control"] in ("H1", "H2")}
        candidatos = [p for p in self.jugadores if p["equipo"] == activo["equipo"]
                      and p is not activo and p["indice"] not in ocupados]
        if not candidatos:
            return
        coincide = next((p for p in candidatos if p is objetivo), None)
        if coincide is not None:
            nuevo = coincide
        else:
            por_indice = sorted(candidatos, key=lambda p: p["indice"])
            nuevo = next((p for p in por_indice if p["indice"] > activo["indice"]), por_indice[0])
        activo["control"] = "IA"
        activo["marcador"].enabled = False
        nuevo["control"] = control
        nuevo["marcador"].enabled = True
        self.mensaje_partido = f"CONTROL · {nuevo['nombre']} · {nuevo['indice'] + 1}"
        self.mensaje_tiempo = 1.0

    def _sincronizar_controles(self) -> None:
        """Sigue la recepción del pase y cambia al defensor más cercano al balón."""
        if self.poseedor is self._poseedor_control_anterior:
            return
        self._poseedor_control_anterior = self.poseedor
        controles = sorted(c for c in ("H1", "H2")
                           if any(p["control"] == c for p in self.jugadores))
        control_del_portador: str | None = None
        poseedor = self.poseedor
        if poseedor is not None:
            portadores_humanos = []
            for control in controles:
                jugador = next((p for p in self.jugadores if p["control"] == control), None)
                if jugador is not None and jugador["equipo"] == poseedor["equipo"]:
                    portadores_humanos.append((control, jugador))
            if portadores_humanos:
                # En cooperativo, solo una persona cambia al receptor del pase.
                # El otro usuario conserva un jugador distinto para dar apoyo.
                control_portador, _ = min(
                    portadores_humanos,
                    key=lambda par: math.hypot(par[1]["e"].x - poseedor["e"].x,
                                               par[1]["e"].z - poseedor["e"].z))
                control_del_portador = control_portador
                self._cambiar_jugador(control_portador, poseedor)
        for control in controles:
            activo = next((p for p in self.jugadores if p["control"] == control), None)
            if activo is None or control == control_del_portador:
                continue
            if self.poseedor is not None and self.poseedor["equipo"] == activo["equipo"]:
                # El segundo usuario mantiene el control de un compañero diferente.
                continue
            if activo["indice"] == 0:
                # Si el usuario tomó al arquero, lo conserva mientras defiende.
                continue
            ocupados = {p["indice"] for p in self.jugadores
                        if p["equipo"] == activo["equipo"] and p is not activo
                        and p["control"] in ("H1", "H2")}
            opciones = [p for p in self.jugadores if p["equipo"] == activo["equipo"]
                        and p["indice"] != 0 and p is not activo
                        and p["indice"] not in ocupados]
            cercano = min(opciones, key=lambda p: math.hypot(
                p["e"].x - self.balon.x, p["e"].z - self.balon.z), default=None)
            if cercano:
                self._cambiar_jugador(control, cercano)

    def _robar(self, control: str) -> None:
        """El usuario disputa la posesión cerca del portador o recoge un balón dividido."""
        p = next((j for j in self.jugadores if j["control"] == control), None)
        if p is None:
            return
        objetivo = self.poseedor
        bx, bz = ((objetivo["e"].x, objetivo["e"].z) if objetivo
                  else (self.balon.x, self.balon.z))
        distancia = math.hypot(p["e"].x - bx, p["e"].z - bz)
        if objetivo is not None and objetivo["equipo"] != p["equipo"] and distancia < 1.35:
            self.poseedor = p
            p["recarga_robo"] = .65
            self.vel_balon = Vec3(0, 0, 0)
        elif objetivo is None and distancia < 1.15:
            self.poseedor = p
            self.vel_balon = Vec3(0, 0, 0)

    def _patear(self, p: dict, accion: str, companero: dict | None = None) -> None:
        if math.hypot(p["e"].x - self.balon.x, p["e"].z - self.balon.z) > 1.55:
            return
        if self.poseedor is not None and self.poseedor["equipo"] != p["equipo"]:
            return
        signo = 1 if p["equipo"] == 0 else -1
        if accion == "pase" and companero:
            dx = companero["e"].x + companero["direccion"][0] * .8 - self.balon.x
            dz = companero["e"].z + companero["direccion"][1] * .8 - self.balon.z
            potencia = 7.2
            altura = 1.4
        elif accion == "centro":
            # Balón bombeado desde el carril hacia la zona de remate.
            meta_x = 9.5 if p["equipo"] == 0 else -9.5
            dx, dz = meta_x - self.balon.x, (companero["e"].z if companero else 0) - self.balon.z
            potencia = 8.1
            altura = 5.2
        elif accion == "pase":
            # Sin compañero cercano, el usuario puede filtrar el balón al espacio.
            dx, dz = p["direccion"]
            potencia = 6.3
            altura = .9
        else:
            teclas = {"H1": ("d", "a", "s", "w"),
                      "H2": ("right arrow", "left arrow", "down arrow", "up arrow")}
            if p["control"] in teclas:
                teclas_presionadas = cast(Any, held_keys)
                derecha, izquierda, abajo, arriba = teclas[p["control"]]
                dx = teclas_presionadas[derecha] - teclas_presionadas[izquierda]
                dz = teclas_presionadas[arriba] - teclas_presionadas[abajo]
                if math.hypot(dx, dz) > 0:
                    largo_aim = math.hypot(dx, dz)
                    p["direccion"] = (dx / largo_aim, dz / largo_aim)
            dx, dz = p["direccion"]
            potencia = 10.5
            altura = 3.3
        largo = max(.001, math.hypot(dx, dz))
        self.vel_balon = Vec3(dx / largo * potencia, altura, dz / largo * potencia)
        self.poseedor = None
        self.ultimo_golpeador = p
        self.audio.efecto("balon")
        self.audio.jugada(p["nombre"], accion)
        if self.ia:
            self.ia.log.action(p["equipo"], p["nombre"], accion, self.tiempo)

    def input(self, key: str) -> None:
        if key == "escape":
            if self.estado == "partido":
                if self.pausa or self.fin:
                    self._volver_al_menu()
                else:
                    self.pausa = True
                    self._actualizar_overlay_partido()
            else:
                self.audio.cerrar()
                application.quit()
        elif self.estado == "menu" and key in ("1", "2", "3", "4"):
            self.iniciar(int(key))
        elif self.estado == "menu" and key in ("a", "d", "left arrow", "right arrow"):
            equipo = 0 if key in ("a", "d") else 1
            paso = 1 if key in ("d", "right arrow") else -1
            self.cambiar_pais(equipo, paso)
        elif self.estado == "menu" and key == "r":
            equipo_ia = 1 if self.modo in (1, 2) else 0
            opciones = [i for i in range(len(SELECCIONES)) if i != self.pais_indices[1 - equipo_ia]]
            self.pais_indices[equipo_ia] = random.choice(opciones)
            self.cambiar_pais(equipo_ia, 0)
        elif key == "v" and self.estado == "partido" and not self.pausa and not self.fin:
            self._alternar_vista()
        elif self.estado == "partido":
            if key == "p" and not self.fin:
                self.pausa = not self.pausa
                self._actualizar_overlay_partido()
            elif key == "r":
                self._reiniciar_partido()
            elif key in ("space", "q", "e", "right control", "enter", "u") and not self.pausa and not self.fin:
                self._accion(key)
            elif key in ("f", "right alt") and not self.pausa and not self.fin:
                self._robar("H1" if key == "f" else "H2")
            elif key == "tab" and not self.pausa and not self.fin:
                self._cambiar_jugador("H1")
            elif key == "backspace" and not self.pausa and not self.fin:
                self._cambiar_jugador("H2")

    # -------------------------------- Bucle ---------------------------------
    def update(self) -> None:
        if self.estado != "partido" or self.pausa or self.fin:
            self._actualizar_hud()
            return
        dt = min(float(cast(Any, time).dt), .04)
        self.tiempo += dt
        self.mensaje_tiempo = max(0.0, self.mensaje_tiempo - dt)
        self.reloj_partido = max(0.0, self.reloj_partido - dt)
        for p in self.jugadores:
            if p["control"] == "IA":
                continue
            dx, dz, sprint = self._movimiento(p["control"])
            magnitud = math.hypot(dx, dz)
            p["moviendo"] = magnitud > 0
            if magnitud:
                dx, dz = dx / magnitud, dz / magnitud
                velocidad = 9.2 if sprint and p["stamina"] > .06 else 7.0
                if self.poseedor is p:
                    velocidad *= .90
                p["e"].x = clamp(p["e"].x + dx * velocidad * dt, -11.4, 11.4)
                p["e"].z = clamp(p["e"].z + dz * velocidad * dt, -6.4, 6.4)
                p["direccion"] = (dx, dz)
                # El modelo mira hacia -Z en reposo; se orienta al vector de movimiento.
                p["e"].rotation_y = math.degrees(math.atan2(-dx, -dz))
                p["stamina"] = max(0.0, p["stamina"] - dt * (.22 if sprint else .07))
            else:
                p["stamina"] = min(1.0, p["stamina"] + dt * .15)

        for p in self.jugadores:
            p["recarga_robo"] = max(0.0, p["recarga_robo"] - dt)
        self._actualizar_ia(dt)
        self._animar_jugadores(dt)
        self._actualizar_posesion(dt)
        self._sincronizar_controles()
        if self.poseedor is not None:
            pais = obtener_seleccion(self.pais_indices[self.poseedor["equipo"]]).nombre
            self.audio.posesion(self.poseedor["nombre"], pais, self.tiempo)
        self._actualizar_balon(dt)
        if self.reloj_partido <= 0:
            self.fin = True
            if self.ia:
                self.ia.log.finish()
        self._actualizar_hud()

    def _actualizar_ia(self, dt: float) -> None:
        if not self.ia:
            return
        self.temporizador_ia -= dt
        if self.temporizador_ia <= 0:
            self.temporizador_ia = .22
            por_equipo = {t: [p for p in self.jugadores
                              if p["equipo"] == t and p["control"] == "IA"] for t in (0, 1)}
            for equipo, miembros in por_equipo.items():
                if not miembros:
                    continue
                signo = 1 if equipo == 0 else -1
                portador = self.poseedor
                rivales = [q for q in self.jugadores if q["equipo"] != equipo]
                cazador = min(miembros, key=lambda q: math.hypot(
                    q["e"].x - self.balon.x, q["e"].z - self.balon.z))
                actor_ia: dict = cazador
                if portador is not None and any(portador is q for q in miembros):
                    actor_ia = portador
                compañero = self._mejor_companero(actor_ia)
                posicion_companero = None
                if compañero is not None:
                    posicion_companero = (
                        compañero["e"].x / 12, compañero["e"].z / 7
                    )
                accion, prediccion = self.ia.decide(
                    (self.balon.x / 12, self.balon.z / 7, clamp(self.balon.y / 3, 0, 1)),
                    (actor_ia["e"].x / 12, actor_ia["e"].z / 7),
                    posicion_companero,
                    signo)
                for p in miembros:
                    if p["indice"] == 0:
                        p["objetivo"] = (-10.8 if equipo == 0 else 10.8,
                                          clamp(self.balon.z * .24, -1.25, 1.25))
                    elif portador is p:
                        meta = 12.0 if equipo == 0 else -12.0
                        distancia_meta = abs(meta - p["e"].x)
                        presion = min((math.hypot(p["e"].x - r["e"].x,
                                                  p["e"].z - r["e"].z) for r in rivales), default=99)
                        distancia_balon = math.hypot(p["e"].x - self.balon.x,
                                                     p["e"].z - self.balon.z)
                        if distancia_balon < 1.55 and distancia_meta < 5.2:
                            self._patear(p, "tiro", compañero)
                        elif distancia_balon < 1.55 and compañero and (
                                (presion < 1.65 and accion == "pase") or
                                (distancia_meta < 8.0 and abs(p["e"].z) > 2.4)):
                            tipo = "centro" if distancia_meta < 8 and abs(p["e"].z) > 2.4 else "pase"
                            self._patear(p, tipo, self._mejor_companero(p, solo_ataque=tipo == "centro"))
                        else:
                            p["objetivo"] = (clamp(p["e"].x + signo * 1.9, -10.8, 10.8),
                                              clamp(p["e"].z * .88, -5.9, 5.9))
                    elif portador is not None and portador["equipo"] != equipo:
                        portador_actual = portador
                        presionador = min(miembros, key=lambda q: math.hypot(
                            q["e"].x - portador_actual["e"].x,
                            q["e"].z - portador_actual["e"].z))
                        if p is presionador:
                            p["objetivo"] = (portador_actual["e"].x, portador_actual["e"].z)
                        else:
                            p["objetivo"] = (clamp(p["casa"][0] + (self.balon.x - p["casa"][0]) * .13,
                                                   -10.8, 10.8),
                                              clamp(p["casa"][1] + (self.balon.z - p["casa"][1]) * .18,
                                                   -5.9, 5.9))
                    elif portador is None and p is cazador:
                        p["objetivo"] = (clamp(prediccion[0] * 12, -10.8, 10.8),
                                          clamp(prediccion[1] * 7, -5.9, 5.9))
                    else:
                        empuje = 2.5 if portador and portador["equipo"] == equipo else -.9
                        p["objetivo"] = (clamp(p["casa"][0] + signo * empuje + self.balon.x * .10,
                                               -10.8, 10.8),
                                          clamp(p["casa"][1] + self.balon.z * .12,
                                               -5.9, 5.9))
        # Separar a los jugadores antes de moverlos evita que se amontonen
        # cuando varios persiguen el balón o llegan al mismo objetivo.
        posiciones = {
            id(jugador): (float(jugador["e"].x), float(jugador["e"].z))
            for jugador in self.jugadores
        }
        for p in self.jugadores:
            if p["control"] != "IA":
                continue
            x, z = posiciones[id(p)]
            dx, dz = p["objetivo"][0] - x, p["objetivo"][1] - z
            distancia = math.hypot(dx, dz)
            vx_objetivo = vz_objetivo = 0.0
            if distancia > .14:
                rapidez_objetivo = min(5.8, distancia * 2.7)
                vx_objetivo = dx / distancia * rapidez_objetivo
                vz_objetivo = dz / distancia * rapidez_objetivo

            separacion_x = separacion_z = 0.0
            for otro in self.jugadores:
                if otro is p:
                    continue
                otro_x, otro_z = posiciones[id(otro)]
                sx, sz = x - otro_x, z - otro_z
                distancia_otro = math.hypot(sx, sz)
                if distancia_otro >= 1.05:
                    continue
                if distancia_otro < .001:
                    diferencia = (
                        p["equipo"] * 7 + p["indice"]
                        - otro["equipo"] * 7 - otro["indice"]
                    )
                    angulo = diferencia * 2.399963229728653
                    ux, uz = math.cos(angulo), math.sin(angulo)
                else:
                    ux, uz = sx / distancia_otro, sz / distancia_otro
                peso = (1.05 - distancia_otro) / 1.05
                separacion_x += ux * peso
                separacion_z += uz * peso

            magnitud_separacion = math.hypot(separacion_x, separacion_z)
            if magnitud_separacion > .001:
                vx_objetivo += separacion_x / magnitud_separacion * 2.0
                vz_objetivo += separacion_z / magnitud_separacion * 2.0
            rapidez_objetivo = math.hypot(vx_objetivo, vz_objetivo)
            if rapidez_objetivo > 5.8:
                factor = 5.8 / rapidez_objetivo
                vx_objetivo *= factor
                vz_objetivo *= factor

            vx, vz = p.get("velocidad_ia", (0.0, 0.0))
            cambio_x, cambio_z = vx_objetivo - vx, vz_objetivo - vz
            magnitud_cambio = math.hypot(cambio_x, cambio_z)
            rapidez_actual = math.hypot(vx, vz)
            aceleracion = 17.0 if rapidez_objetivo > rapidez_actual else 21.0
            cambio_maximo = aceleracion * max(0.0, dt)
            if magnitud_cambio > cambio_maximo > 0:
                factor = cambio_maximo / magnitud_cambio
                cambio_x *= factor
                cambio_z *= factor
            vx += cambio_x
            vz += cambio_z

            siguiente_x = clamp(x + vx * dt, -11.4, 11.4)
            siguiente_z = clamp(z + vz * dt, -6.4, 6.4)
            if siguiente_x != x + vx * dt:
                vx = 0.0
            if siguiente_z != z + vz * dt:
                vz = 0.0
            p["e"].x, p["e"].z = siguiente_x, siguiente_z
            p["velocidad_ia"] = (vx, vz)

            rapidez_actual = math.hypot(vx, vz)
            p["moviendo"] = rapidez_actual > .12
            if rapidez_actual > .08:
                nx, nz = vx / rapidez_actual, vz / rapidez_actual
                p["direccion"] = (nx, nz)
                giro_objetivo = math.degrees(math.atan2(-nx, -nz))
                giro_actual = float(p["e"].rotation_y)
                diferencia_giro = (giro_objetivo - giro_actual + 180) % 360 - 180
                giro_maximo = 540.0 * max(0.0, dt)
                p["e"].rotation_y = giro_actual + clamp(
                    diferencia_giro, -giro_maximo, giro_maximo
                )

    def _actualizar_posesion(self, dt: float) -> None:
        """Pega el balón al portador y permite intercepciones y recuperaciones."""
        if self.poseedor is not None:
            p = self.poseedor
            dx, dz = p["direccion"]
            self.balon.position = Vec3(p["e"].x + dx * .72, RADIO_BALON, p["e"].z + dz * .72)
            self.vel_balon = Vec3(0, 0, 0)
            # La defensa de la máquina intenta recuperar el balón al entrar en rango.
            for defensor in self.jugadores:
                if (defensor["control"] == "IA" and defensor["equipo"] != p["equipo"]
                        and defensor["recarga_robo"] <= 0 and p["recarga_robo"] <= 0
                        and math.hypot(defensor["e"].x - p["e"].x,
                                       defensor["e"].z - p["e"].z) < .78):
                    self.poseedor = defensor
                    defensor["recarga_robo"] = .8
                    break
            if self.poseedor is not None:
                return

        velocidad = math.hypot(self.vel_balon.x, self.vel_balon.z)
        receptor = min(self.jugadores, key=lambda q: math.hypot(
            q["e"].x - self.balon.x, q["e"].z - self.balon.z), default=None)
        umbral = .78 if velocidad < 5.0 else .48
        if (receptor and self.balon.y < 1.25
                and math.hypot(receptor["e"].x - self.balon.x,
                               receptor["e"].z - self.balon.z) < umbral):
            self.poseedor = receptor
            self.vel_balon = Vec3(0, 0, 0)

    def _actualizar_balon(self, dt: float) -> None:
        if self.poseedor is not None:
            return
        self.vel_balon.y -= 9.8 * dt
        self.balon.position += self.vel_balon * dt
        self.balon.rotation_x += self.vel_balon.z * dt * 70
        self.balon.rotation_z -= self.vel_balon.x * dt * 70
        if self.balon.y < RADIO_BALON:
            self.balon.y = RADIO_BALON
            self.vel_balon.y = abs(self.vel_balon.y) * .40
            self.vel_balon.x *= .985
            self.vel_balon.z *= .985
        if abs(self.balon.z) > 6.75:
            self.balon.z = clamp(self.balon.z, -6.75, 6.75)
            self.vel_balon.z *= -.62
        if abs(self.balon.x) > 12.35:
            if abs(self.balon.z) < 1.72 and self.balon.y < 2.05:
                equipo = 0 if self.balon.x > 0 else 1
                autor = self.ultimo_golpeador
                if self.ia:
                    self.ia.log.goal(equipo, autor["nombre"] if autor else "Equipo", self.tiempo)
                    pais_goleador = obtener_seleccion(self.pais_indices[equipo]).nombre
                    self.audio.gol(autor["nombre"] if autor else "el equipo", pais_goleador)
                    if max(self.ia.log.score.values()) >= 5:
                        self.fin = True
                        self.ia.log.finish()
                equipo_saque = 1 - equipo
                self._iniciar_saque(equipo_saque)
                self.mensaje_partido = (
                    f"¡GOL {obtener_seleccion(self.pais_indices[equipo]).nombre.upper()}!  ·  "
                    f"SAQUE {obtener_seleccion(self.pais_indices[equipo_saque]).nombre.upper()}"
                )
                self.mensaje_tiempo = 2.4
            else:
                self.balon.x = math.copysign(12.35, self.balon.x)
                self.vel_balon.x *= -.62

    def _actualizar_hud(self) -> None:
        if not self.hud_widgets or not self.ia:
            return
        minutos, segundos = divmod(int(self.reloj_partido), 60)
        pais0 = obtener_seleccion(self.pais_indices[0])
        pais1 = obtener_seleccion(self.pais_indices[1])
        if self.hud_marcador is not None:
            self.hud_marcador.text = (
                f"{self.ia.log.score[0]}  -  {self.ia.log.score[1]}"
            )
        if self.hud_reloj is not None:
            self.hud_reloj.text = f"{minutos:02}:{segundos:02}"
        if self.panel_marcador is not None:
            self.panel_marcador.text = (f"{pais0.sigla}  {self.ia.log.score[0]}  —  "
                                        f"{self.ia.log.score[1]}  {pais1.sigla}")
        posesion = (obtener_seleccion(self.pais_indices[self.poseedor["equipo"]]).sigla
                    if self.poseedor else "DIVIDIDA")
        if self.hud_estado is not None:
            if self.mensaje_tiempo > 0 and self.mensaje_partido:
                self.hud_estado.text = self.mensaje_partido
            elif self.pausa:
                self.hud_estado.text = "PARTIDO EN PAUSA"
            elif self.fin:
                self.hud_estado.text = "FINAL"
            else:
                nombre = f" · {self.poseedor['nombre']}" if self.poseedor else ""
                self.hud_estado.text = f"POSESIÓN {posesion}{nombre}"

        for control, equipo in (("H1", 0), ("H2", 1)):
            etiqueta = self.hud_nombres.get(control)
            if etiqueta is None:
                continue
            jugador = next(
                (p for p in self.jugadores if p["control"] == control), None
            )
            if jugador is None:
                candidatos = [p for p in self.jugadores if p["equipo"] == equipo]
                jugador = min(
                    candidatos,
                    key=lambda p: distancia_horizontal(
                        p["e"].x - self.balon.x, p["e"].z - self.balon.z
                    ),
                    default=None,
                )
            sigla = obtener_seleccion(self.pais_indices[equipo]).sigla
            etiqueta.text = f"{sigla}  ·  {jugador['nombre']}" if jugador else sigla

        radar_x, radar_y, radar_w, radar_h = 0.0, -.365, .265, .145
        for jugador, punto in zip(self.jugadores, self.radar_puntos):
            punto.x = radar_x + max(-.48, min(.48, jugador["e"].x / (LARGO / 2))) * radar_w / 2
            punto.y = radar_y + max(-.46, min(.46, jugador["e"].z / (ANCHO / 2))) * radar_h / 2
        if self.radar_balon is not None:
            self.radar_balon.x = radar_x + max(
                -.48, min(.48, self.balon.x / (LARGO / 2))
            ) * radar_w / 2
            self.radar_balon.y = radar_y + max(
                -.46, min(.46, self.balon.z / (ANCHO / 2))
            ) * radar_h / 2
        if self.pausa:
            if self.hud_estado is not None:
                self.hud_estado.text = "PAUSA  ·  P CONTINÚA  ·  R REINICIA"
        elif self.fin:
            if self.hud_estado is not None:
                self.hud_estado.text = "FINAL  ·  R REINICIA"
        self._actualizar_overlay_partido()

Juego3DBase = Juego3D

# ===== Modelos, animaciones y estadio volumétricos =====

RADIO_BALON = .21
RADIO_POSTE_MAS_BALON = .32
RADIO_TRAVESAÑO_MAS_BALON = .32


def producto(a: Vec3, b: Vec3) -> float:
    return float(a.x * b.x + a.y * b.y + a.z * b.z)


def distancia_horizontal(x: float, z: float) -> float:
    return math.hypot(x, z)


class Juego3DVolumetrico(Juego3DBase):
    """Amplía el partido existente con atletas, graderías y físicas volumétricas."""

    def __init__(self) -> None:
        self.cancha_de_barrio = ESCENARIO_3D == "barrio"
        self._cuenta_gradas: dict[str, int] = {}
        super().__init__()
        # La retransmisión usa futbolistas geométricos de baja poligonización,
        # como en los juegos de fútbol de PS1; la lógica usa los mismos actores.

    def _multitud(self) -> None:
        """La grada volumétrica se genera junto con sus asientos y sectores."""
        return

    def _crear_jugadores(self) -> None:
        super()._crear_jugadores()
        for jugador in self.jugadores:
            jugador["velocidad_anim"] = 0.0
            jugador["animador"] = AnimadorProcedural(jugador)
            if jugador["indice"] == 0:
                x_guardameta = -11.25 if jugador["equipo"] == 0 else 11.25
                jugador["e"].x = x_guardameta
                jugador["casa"] = (x_guardameta, jugador["casa"][1])
                jugador["objetivo"] = jugador["casa"]

    def _actualizar_ia(self, dt: float) -> None:
        """Conserva la IA existente y da funciones individuales al modo 4."""
        posiciones = {
            id(j): (float(j["e"].x), float(j["e"].z)) for j in self.jugadores
        }
        super()._actualizar_ia(dt)

        if self.modo == 4:
            # Dos atacantes de la PC se coordinan por separado: uno presiona o
            # desmarca y otro apoya o cubre al rival con el balón.
            atacantes = [
                p for p in self.jugadores
                if p["equipo"] == 0 and p["indice"] in (5, 6)
                and p["control"] == "IA"
            ]
            portador = self.poseedor
            if portador is None:
                cazador = min(
                    atacantes,
                    key=lambda p: distancia_horizontal(
                        p["e"].x - self.balon.x, p["e"].z - self.balon.z
                    ),
                    default=None,
                )
                for p in atacantes:
                    if p is cazador:
                        p["objetivo"] = (
                            float(self.balon.x + self.vel_balon.x * .16),
                            float(self.balon.z + self.vel_balon.z * .16),
                        )
                    else:
                        p["objetivo"] = (
                            float(self.balon.x - 1.5), float(-self.balon.z * .65)
                        )
            elif portador["equipo"] == 0:
                for p in atacantes:
                    if p is portador:
                        p["objetivo"] = (float(p["e"].x), float(p["e"].z))
                    elif p["indice"] == 6:
                        p["objetivo"] = (
                            min(10.6, float(portador["e"].x + 3.0)),
                            float(portador["e"].z * .55),
                        )
                    else:
                        p["objetivo"] = (
                            float(portador["e"].x - 1.0),
                            float(-portador["e"].z * .8),
                        )
            else:
                rivales = [
                    p for p in self.jugadores
                    if p["equipo"] == 1 and p["indice"] != 0
                ]
                marcado = min(
                    rivales,
                    key=lambda p: distancia_horizontal(
                        p["e"].x - portador["e"].x,
                        p["e"].z - portador["e"].z,
                    ),
                    default=None,
                )
                for p in atacantes:
                    objetivo = marcado if p["indice"] == 5 else portador
                    if objetivo:
                        p["objetivo"] = (
                            float(objetivo["e"].x), float(objetivo["e"].z)
                        )

        for portero in self.jugadores:
            if portero["indice"] != 0:
                continue
            if portero["control"] == "IA":
                x_meta = -11.25 if portero["equipo"] == 0 else 11.25
                z_meta = max(-1.42, min(1.42, float(self.balon.z) * .43))
                portero["objetivo"] = (x_meta, z_meta)
                paso = min(1.0, dt * 7.0)
                portero["e"].x += (x_meta - portero["e"].x) * paso
                portero["e"].z += (z_meta - portero["e"].z) * paso

                se_acerca = (
                    self.vel_balon.x < -4.0 if portero["equipo"] == 0
                    else self.vel_balon.x > 4.0
                )
                if (
                    self.poseedor is None and se_acerca
                    and abs(float(portero["e"].x - self.balon.x)) < 4.0
                    and self.balon.y < 2.2
                    and abs(float(self.balon.z - portero["e"].z)) < 2.1
                    and portero["animador"].tiempo_restante <= 0
                ):
                    diferencia = float(self.balon.z - portero["e"].z)
                    estado = "portero_salto" if self.balon.y > 1.0 else "portero_estirada"
                    portero["animador"].activar(
                        estado, .62, direccion=1 if diferencia >= 0 else -1
                    )

        if dt > 0:
            for jugador in self.jugadores:
                anterior = posiciones.get(id(jugador))
                if anterior is not None:
                    jugador["velocidad_anim"] = distancia_horizontal(
                        float(jugador["e"].x) - anterior[0],
                        float(jugador["e"].z) - anterior[1],
                    ) / dt

    def _animar_jugadores(self, dt: float) -> None:
        """Anima las articulaciones de los modelos 3D en ambas cámaras."""
        super()._animar_jugadores(dt)
        for jugador in self.jugadores:
            jugador["animador"].actualizar(jugador, dt)

    def _patear(self, jugador: dict, accion: str, companero: dict | None = None) -> None:
        super()._patear(jugador, accion, companero)
        if self.ultimo_golpeador is jugador:
            animacion = {"pase": "pase", "centro": "centro", "tiro": "disparo"}.get(
                accion, "disparo"
            )
            jugador["animador"].activar(
                animacion, .46 if accion == "tiro" else .38
            )

    def _iniciar_saque(self, equipo: int) -> None:
        super()._iniciar_saque(equipo)
        for jugador in self.jugadores:
            if jugador.get("animador"):
                jugador["animador"].activar("espera", .01)

    def _colisionar_con_porteros(self, anterior: Vec3, siguiente: Vec3) -> bool:
        """Resuelve el contacto del balón con el cuerpo geométrico del portero."""
        movimiento = cast(Vec3, siguiente - anterior)
        plano = Vec3(movimiento.x, 0, movimiento.z)
        longitud2 = max(.00001, producto(plano, plano))
        for portero in self.jugadores:
            if portero["indice"] != 0:
                continue
            centro = cast(Vec3, portero["e"].position + Vec3(0, .93, 0))
            origen_plano = Vec3(anterior.x, 0, anterior.z)
            centro_plano = Vec3(centro.x, 0, centro.z)
            desplazamiento_centro = cast(Vec3, centro_plano - origen_plano)
            t = max(
                0.0,
                min(1.0, producto(desplazamiento_centro, plano) / longitud2),
            )
            cercano = cast(Vec3, anterior + movimiento * t)
            diferencia = Vec3(cercano.x - centro.x, 0, cercano.z - centro.z)
            distancia = math.sqrt(max(0.0, producto(diferencia, diferencia)))
            acercamiento = producto(self.vel_balon, diferencia)
            if (
                distancia >= .69 or not .21 <= cercano.y <= 1.94
                or (distancia > .0001 and acercamiento >= 0)
            ):
                continue
            normal = (
                Vec3(diferencia.x / distancia, 0, diferencia.z / distancia)
                if distancia > .0001
                else Vec3(-1 if self.vel_balon.x > 0 else 1, 0, 0)
            )
            siguiente += normal * (.69 - distancia + .015)
            componente = producto(self.vel_balon, normal)
            self.vel_balon -= normal * (1.55 * componente)
            self.vel_balon *= .72
            estado = "portero_salto" if cercano.y > 1.0 else "portero_estirada"
            portero["animador"].activar(
                estado, .62, direccion=1 if self.balon.z >= portero["e"].z else -1
            )
            return True
        return False

    def _colisionar_con_marcos(self, anterior: Vec3, siguiente: Vec3) -> bool:
        """Prueba el segmento de vuelo contra postes y travesaños de ambas metas."""
        movimiento = cast(Vec3, siguiente - anterior)
        hubo_contacto = False
        for lado in (-1, 1):
            x_meta = lado * 12.35

            # Postes verticales: distancia del segmento del balón a una cápsula.
            movimiento_plano = Vec3(movimiento.x, 0, movimiento.z)
            longitud2 = max(.00001, producto(movimiento_plano, movimiento_plano))
            for z_poste in (-1.70, 1.70):
                centro = Vec3(x_meta, 0, z_poste)
                origen = Vec3(anterior.x, 0, anterior.z)
                desplazamiento_centro = cast(Vec3, centro - origen)
                t = max(
                    0.0,
                    min(1.0, producto(desplazamiento_centro, movimiento_plano) / longitud2),
                )
                punto = cast(Vec3, anterior + movimiento * t)
                eje = Vec3(x_meta, max(.05, min(1.85, punto.y)), z_poste)
                diferencia = cast(Vec3, punto - eje)
                distancia = math.sqrt(max(0.0, producto(diferencia, diferencia)))
                if distancia < RADIO_POSTE_MAS_BALON:
                    normal = (
                        Vec3(diferencia.x / distancia,
                             diferencia.y / distancia,
                             diferencia.z / distancia) if distancia > .0001
                        else Vec3(-1 if self.vel_balon.x > 0 else 1, 0, 0)
                    )
                    siguiente += normal * (RADIO_POSTE_MAS_BALON - distancia + .01)
                    componente = producto(self.vel_balon, normal)
                    if componente < 0:
                        self.vel_balon -= normal * (1.55 * componente)
                    self.vel_balon *= .76
                    hubo_contacto = True

            # Travesaño horizontal, limitado al tramo entre los postes.
            movimiento_xy = Vec3(movimiento.x, movimiento.y, 0)
            longitud_xy2 = max(.00001, producto(movimiento_xy, movimiento_xy))
            hacia_eje = Vec3(x_meta - anterior.x, 1.90 - anterior.y, 0)
            t_barra = max(
                0.0, min(1.0, producto(hacia_eje, movimiento_xy) / longitud_xy2)
            )
            punto_barra = cast(Vec3, anterior + movimiento * t_barra)
            z_barra = max(-1.75, min(1.75, float(punto_barra.z)))
            eje_barra = Vec3(x_meta, 1.90, z_barra)
            diferencia_barra = cast(Vec3, punto_barra - eje_barra)
            distancia_barra = math.sqrt(
                max(0.0, producto(diferencia_barra, diferencia_barra))
            )
            if distancia_barra < RADIO_TRAVESAÑO_MAS_BALON:
                normal = (
                    Vec3(diferencia_barra.x / distancia_barra,
                         diferencia_barra.y / distancia_barra,
                         diferencia_barra.z / distancia_barra)
                    if distancia_barra > .0001
                    else Vec3(-1 if self.vel_balon.x > 0 else 1, 0, 0)
                )
                siguiente += normal * (
                    RADIO_TRAVESAÑO_MAS_BALON - distancia_barra + .01
                )
                componente = producto(self.vel_balon, normal)
                if componente < 0:
                    self.vel_balon -= normal * (1.55 * componente)
                self.vel_balon *= .76
                hubo_contacto = True
        return hubo_contacto

    def _registrar_gol(self, equipo: int) -> None:
        autor = self.ultimo_golpeador
        if self.ia:
            self.ia.log.goal(
                equipo, autor["nombre"] if autor else "Equipo", self.tiempo
            )
            pais = self.obtener_nombre_pais(self.pais_indices[equipo])
            self.audio.gol(autor["nombre"] if autor else "el equipo", pais)
            if max(self.ia.log.score.values()) >= 5:
                self.fin = True
                self.ia.log.finish()

        saque = 1 - equipo
        nombre_goleador = self.obtener_nombre_pais(self.pais_indices[equipo])
        nombre_saque = self.obtener_nombre_pais(self.pais_indices[saque])
        self._iniciar_saque(saque)
        self.mensaje_partido = (
            f"¡GOL {nombre_goleador.upper()}!  ·  SAQUE {nombre_saque.upper()}"
        )
        self.mensaje_tiempo = 2.4
        if autor and autor.get("animador"):
            autor["animador"].activar("celebracion", 1.7)

    @staticmethod
    def obtener_nombre_pais(indice: int) -> str:

        return obtener_seleccion(indice).nombre

    def _reponer_por_salida(self, tipo: str, lado: int, z_salida: float) -> None:
        """Reanuda con banda, saque de meta o tiro de esquina."""
        ultimo = self.ultimo_golpeador
        equipo_ultimo = ultimo["equipo"] if ultimo else None
        nombre_tipo = "SAQUE DE BANDA"
        x_balon = max(-11.45, min(11.45, float(self.balon.x)))
        z_balon = math.copysign(ANCHO / 2 - .55, z_salida or self.vel_balon.z or 1)

        if tipo == "meta":
            equipo_defensor = 1 if lado > 0 else 0
            equipo_atacante = 1 - equipo_defensor
            if equipo_ultimo == equipo_atacante:
                equipo_saque = equipo_defensor
                jugador = next((p for p in self.jugadores
                                if p["equipo"] == equipo_saque and p["indice"] == 0), None)
                nombre_tipo = "SAQUE DE META"
                x_balon, z_balon = lado * 9.65, 0.0
                direccion = (-float(lado), 0.0)
            else:
                equipo_saque = equipo_atacante
                z_balon = math.copysign(ANCHO / 2 - .55, z_salida or 1)
                x_balon = lado * (LARGO / 2 - 1.25)
                nombre_tipo = "TIRO DE ESQUINA"
                jugador = min(
                    (p for p in self.jugadores
                     if p["equipo"] == equipo_saque and p["indice"] != 0),
                    key=lambda p: math.hypot(p["e"].x - x_balon, p["e"].z - z_balon),
                    default=None,
                )
                direccion = (-float(lado), -math.copysign(.2, z_balon))
        else:
            equipo_saque = (1 - equipo_ultimo if equipo_ultimo is not None
                            else (0 if self.vel_balon.x >= 0 else 1))
            x_balon = max(-11.45, min(11.45, float(self.balon.x)))
            jugador = min(
                (p for p in self.jugadores
                 if p["equipo"] == equipo_saque and p["indice"] != 0),
                key=lambda p: math.hypot(p["e"].x - x_balon, p["e"].z - z_balon),
                default=None,
            )
            direccion = (0.0, -math.copysign(1.0, z_balon))

        if jugador is None:
            return
        jugador["direccion"] = direccion
        jugador["e"].position = Vec3(x_balon - direccion[0] * .72, 0, z_balon - direccion[1] * .72)
        jugador["e"].x = max(-11.4, min(11.4, float(jugador["e"].x)))
        jugador["e"].z = max(-6.4, min(6.4, float(jugador["e"].z)))
        jugador["objetivo"] = (float(jugador["e"].x), float(jugador["e"].z))
        jugador["moviendo"] = False
        jugador["recarga_robo"] = .9
        jugador["e"].rotation_y = math.degrees(math.atan2(-direccion[0], -direccion[1]))
        self.poseedor = jugador
        self.balon.position = Vec3(jugador["e"].x + direccion[0] * .72, RADIO_BALON, jugador["e"].z + direccion[1] * .72)
        self.vel_balon = Vec3(0, 0, 0)
        self._poseedor_control_anterior = None
        self.mensaje_partido = f"{nombre_tipo}  /  {self.obtener_nombre_pais(equipo_saque).upper()}"
        self.mensaje_tiempo = 2.0
        self.audio.efecto("silbato")

    def _actualizar_balon(self, dt: float) -> None:
        """Integra la esfera y resuelve rebotes, atajadas, postes y goles."""
        if self.poseedor is not None:
            return

        anterior = Vec3(self.balon.x, self.balon.y, self.balon.z)
        self.vel_balon.y -= 9.8 * dt
        siguiente = cast(Vec3, anterior + self.vel_balon * dt)
        self.balon.rotation_x += self.vel_balon.z * dt * 70
        self.balon.rotation_z -= self.vel_balon.x * dt * 70

        if siguiente.y <= RADIO_BALON:
            siguiente.y = RADIO_BALON
            self.vel_balon.y = abs(self.vel_balon.y) * .39
            if self.vel_balon.y < .85:
                self.vel_balon.y = 0
            rozamiento = math.exp(-.36 * dt)
            self.vel_balon.x *= rozamiento
            self.vel_balon.z *= rozamiento
        else:
            rozamiento = math.exp(-.035 * dt)
            self.vel_balon.x *= rozamiento
            self.vel_balon.z *= rozamiento

        self._colisionar_con_porteros(anterior, siguiente)
        self._colisionar_con_marcos(anterior, siguiente)

        for lado in (-1, 1):
            x_meta = lado * 12.35
            cruza = (
                anterior.x < x_meta <= siguiente.x if lado == 1
                else anterior.x > x_meta >= siguiente.x
            )
            if not cruza:
                continue
            delta_x = float(siguiente.x - anterior.x)
            if abs(delta_x) < .00001:
                continue
            fraccion = (x_meta - anterior.x) / delta_x
            y_cruce = anterior.y + (siguiente.y - anterior.y) * fraccion
            z_cruce = anterior.z + (siguiente.z - anterior.z) * fraccion
            if RADIO_BALON <= y_cruce <= 1.60 and abs(z_cruce) <= 1.40:
                self._registrar_gol(0 if lado == 1 else 1)
                return
            self._reponer_por_salida("meta", lado, float(z_cruce))
            return

        limite_banda = ANCHO / 2 + RADIO_BALON
        if abs(siguiente.z) >= limite_banda:
            self._reponer_por_salida("banda", 0, float(siguiente.z))
            return

        if abs(siguiente.x) > 12.35:
            lado = 1 if siguiente.x > 0 else -1
            self._reponer_por_salida("meta", lado, float(siguiente.z))
            return

        self.balon.position = siguiente
        if (
            abs(self.vel_balon.x) + abs(self.vel_balon.y) + abs(self.vel_balon.z)
            < .06
        ):
            self.vel_balon = Vec3(0, 0, 0)

    def _seguir_camara(self, dt: float) -> None:
        """Camara de transmision retro, con anticipacion suave de la jugada."""
        if self.vista_cenital or self.estado != "partido":
            return
        dt = max(0.0, min(float(dt), .06))
        velocidad = distancia_horizontal(self.vel_balon.x, self.vel_balon.z)
        anticipacion = min(.28, velocidad * .018)
        x_objetivo = float(self.balon.x) + self.vel_balon.x * anticipacion
        z_objetivo = float(self.balon.z) + self.vel_balon.z * anticipacion
        foco = Vec3(
            max(-2.4, min(2.4, x_objetivo * .19)),
            0,
            max(-.9, min(.9, z_objetivo * .13)),
        )
        distancia = 24.5 - min(1.2, velocidad * .04)
        posicion_objetivo = cast(
            Vec3, foco + Vec3(0, distancia * .96, -distancia)
        )
        mezcla = 1.0 - math.exp(-2.0 * dt)
        posicion_actual = cast(Vec3, camera.position)
        desplazamiento = cast(Vec3, posicion_objetivo - posicion_actual)
        camera.position = cast(
            Vec3, posicion_actual + desplazamiento * mezcla
        )
        camera.look_at(foco)
        camara_ursina = cast(Any, camera)
        fov_actual = float(camara_ursina.fov)
        camara_ursina.fov = fov_actual + (50.0 - fov_actual) * min(1.0, dt * .8)

    def update(self) -> None:
        super().update()
        self._seguir_camara(min(float(cast(Any, time).dt), .04))

    def _configurar_camara(self) -> None:
        super()._configurar_camara()
        if not self.vista_cenital:
            camera.look_at(Vec3(0, 0, 0))

    def cerrar(self) -> None:
        try:
            self.audio.cerrar()
        except (AttributeError, RuntimeError):
            pass


def comprobar_modos_controles_y_gol(juego: Juego3DVolumetrico) -> None:
    """Comprueba escena, cuatro formatos, controles, tiro, atajada y gol."""
    for modo in (1, 2, 3, 4):
        juego.iniciar(modo)
        ia = juego.ia
        assert ia is not None
        humanos = [p for p in juego.jugadores if p["control"] in ("H1", "H2")]
        total_humanos = 1 if modo == 1 else 2
        assert len(juego.jugadores) == 14
        assert all(
            sum(p["equipo"] == equipo for p in juego.jugadores) == 7
            for equipo in (0, 1)
        ), "cada equipo debe tener siete jugadores"
        assert len(humanos) == total_humanos
        assert all(
            p["cuerpo"] is not None and p["cuerpo"].enabled
            and p["sombra"] is not None and p["sombra"].enabled
            for p in juego.jugadores if p["indice"] != 0
        ), "la transmision debe mostrar los modelos 3D y las sombras de los jugadores"
        juego._actualizar_hud()
        assert len(juego.radar_puntos) == 14 and juego.radar_balon is not None
        ia.log.done = True  # No guardar partidas simuladas por esta prueba.

    ia = juego.ia
    assert ia is not None
    registro = ia.log
    teclas = cast(Any, held_keys)
    teclas["w"] = 1
    assert juego._movimiento("H1")[1] > 0
    teclas["w"] = 0
    teclas["up arrow"] = 1
    assert juego._movimiento("H2")[1] > 0
    teclas["up arrow"] = 0

    # En modo 4, el usuario 1 es del equipo visitante y ataca hacia -X.
    jugador = next(p for p in juego.jugadores if p["control"] == "H1")
    jugador["e"].position = Vec3(-10.0, 0, 0)
    jugador["direccion"] = (-1.0, 0.0)
    juego.balon.position = Vec3(-10.55, .42, 0)
    juego.poseedor = jugador
    juego._patear(jugador, "tiro")
    assert juego.poseedor is None and juego.vel_balon.x < 0
    assert jugador["animador"].estado == "disparo"

    # La trayectoria contra el arquero vuelve al campo y no cuenta como gol.
    juego.poseedor = None
    juego.balon.position = Vec3(-11.0, .90, 0)
    juego.vel_balon = Vec3(-10.0, 0, 0)
    marcador_antes = registro.score[1]
    juego._actualizar_balon(.04)
    assert juego.vel_balon.x > 0, "el portero no desvió el balón"
    assert registro.score[1] == marcador_antes

    # Un disparo que toca el poste y otro que supera el larguero no son goles.
    juego.poseedor = None
    juego.balon.position = Vec3(-12.20, 1.15, 1.50)
    juego.vel_balon = Vec3(-10.0, 0, 0)
    juego._actualizar_balon(.04)
    assert registro.score[1] == marcador_antes, "un tiro al poste contó como gol"
    juego.poseedor = None
    juego.balon.position = Vec3(-12.20, 1.82, 0)
    juego.vel_balon = Vec3(-10.0, 0, 0)
    juego._actualizar_balon(.04)
    assert registro.score[1] == marcador_antes, "un tiro alto contó como gol"

    # Un tiro centrado y bajo atraviesa el hueco y devuelve el saque al rival.
    juego.poseedor = None
    juego.balon.position = Vec3(-12.20, .70, 0)
    juego.vel_balon = Vec3(-10.0, 0, 0)
    juego._actualizar_balon(.04)
    assert registro.score[1] == marcador_antes + 1
    assert juego.balon.x == 0

    juego.poseedor = None
    juego.ultimo_golpeador = next(
        p for p in juego.jugadores if p["equipo"] == 0 and p["indice"] == 6
    )
    juego.balon.position = Vec3(0, RADIO_BALON, 6.98)
    juego.vel_balon = Vec3(0, 0, 10)
    juego._actualizar_balon(.04)
    assert juego.poseedor and juego.poseedor["equipo"] == 1
    assert "BANDA" in juego.mensaje_partido

    juego.poseedor = None
    juego.ultimo_golpeador = next(
        p for p in juego.jugadores if p["equipo"] == 0 and p["indice"] == 6
    )
    juego.balon.position = Vec3(12.20, .70, 3.0)
    juego.vel_balon = Vec3(10, 0, 0)
    juego._actualizar_balon(.04)
    assert juego.poseedor and juego.poseedor["equipo"] == 1
    assert juego.poseedor["indice"] == 0
    assert "META" in juego.mensaje_partido

    juego.poseedor = None
    juego.ultimo_golpeador = next(
        p for p in juego.jugadores if p["equipo"] == 1 and p["indice"] == 6
    )
    juego.balon.position = Vec3(12.20, .70, 3.0)
    juego.vel_balon = Vec3(10, 0, 0)
    juego._actualizar_balon(.04)
    assert juego.poseedor and juego.poseedor["equipo"] == 0
    assert "ESQUINA" in juego.mensaje_partido
    registro.done = True
    print(
        "PRUEBA OK: cuatro modos, 14 futbolistas 3D (7 por equipo), controles, "
        "tiro, atajada, gol, saque central, banda, meta y esquina."
    )
    sys.stdout.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="Fútbol 3D volumétrico con Ursina")
    parser.add_argument(
        "--prueba-rapida", action="store_true",
        help="abre una escena breve y comprueba los cuatro modos y la física",
    )
    args = parser.parse_args()

    app = cast(Any, Ursina(
        title="Retro Cup 2002 - Futbol 3D",
        fullscreen=False,
        development_mode=False,
        editor_ui_enabled=False,
    ))
    monitor = window.main_monitor
    if monitor:
        # El formato 4:3 refuerza el encuadre clásico de PlayStation.
        alto = min(900, int(monitor.height), int(monitor.width * .75))
        ancho = int(alto * 4 / 3)
        window.size = (ancho, alto)
        window.position = (
            int(monitor.x + (monitor.width - ancho) / 2),
            int(monitor.y + (monitor.height - alto) / 2),
        )
    juego = Juego3DVolumetrico()

    if args.prueba_rapida:
        comprobar_modos_controles_y_gol(juego)
        invoke(juego.cerrar)
        invoke(application.quit, delay=2.0)

    app.run()


if __name__ == "__main__":
    main()
