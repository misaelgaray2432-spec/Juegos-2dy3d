# PITCH · Fútbol 2D y 3D con selecciones e IA

Dos juegos independientes y autocontenidos: juego_2d.py (Pygame) y juego_3d.py (Ursina). Cada uno incluye la IA,
las estadísticas Pandas, el audio, las selecciones y sus recursos visuales. El código anterior se conserva en respaldo_codigo_pre_fusion_20261014.zip. Los equipos tienen 11 jugadores
En 2D se conserva la plantilla de 11 contra 11; en 3D se juega 7 contra 7.
Los compañeros sin control humano se mueven con la IA. El portero queda en
manos de la IA por defecto, pero puede seleccionarse manualmente.

## Python y dependencias

Requiere Windows de 64 bits y Python 3.12 de 64 bits. El entorno `.2d3d` no va
dentro del ZIP: se instala por separado.

```powershell
& C:\OCT13\.2d3d\Scripts\python.exe -m pip install --upgrade pip
& C:\OCT13\.2d3d\Scripts\python.exe -m pip install -r C:\OCT13\Futbol_IA_2D_3D\requirements.txt
& C:\OCT13\.2d3d\Scripts\python.exe -c "import sys; print(sys.executable); print(sys.version)"
```

La última orden debe mostrar `.2d3d` y Python 3.12. Para abrir los juegos:

```powershell
Set-Location C:\OCT13\Futbol_IA_2D_3D
& C:\OCT13\.2d3d\Scripts\python.exe .\juego_2d.py
# En otra ejecución, para el juego tridimensional:
& C:\OCT13\.2d3d\Scripts\python.exe .\juego_3d.py
```

## Control de los jugadores

Como en un juego de fútbol de consola, TAB (J1) o Retroceso/Backspace (J2)
cambia entre los jugadores disponibles de su equipo, incluido el portero
(11 en 2D y 7 en 3D). El cambio automático sigue al receptor de un pase y al defensor más próximo cuando toca
recuperar la pelota. En cooperativo cada persona conserva un jugador diferente.

- J1: WASD para moverse, Espacio para tirar, Q pase, E centro, F robo y Shift
  izquierdo para correr; TAB cambia el jugador activo.
- J2: flechas para moverse, Ctrl derecho tiro, Enter pase, U centro, Alt derecho
  robo y Shift derecho para correr; Retroceso/Backspace cambia el jugador activo.
- En 3D, V alterna la cámara de transmisión y la cenital. P o Esc abre la
  pausa; allí puedes continuar, reiniciar o volver a la selección. R reinicia
  el encuentro en cualquier momento. Esc desde la pausa vuelve al menú.

Los modos 1 a 4 están disponibles en cada juego: usuario contra IA, duelo local,
dos usuarios cooperando contra IA y dos usuarios contra dos jugadores de IA.
El juego 2D conserva el formato 11 contra 11; el 3D usa siete por equipo.

## Audio y narrador

`assets/audio/` incluye sonidos WAV de estadio, gol, silbato y balón; no hace
falta descargarlos. El comentarista anuncia la posesión, los pases, los centros,
los tiros y los goles mediante la voz local de Windows (pyttsx3/SAPI, con
System.Speech como respaldo). No usa internet. Si no se oye, comprueba el volumen
de Windows y revisa `assets/narrador_estado.txt` después de abrir un partido;
una voz española instalada mejora la pronunciación.

Pandas guarda historial y estadísticas en `data/`. Scikit-learn elige acciones
y Keras predice apoyos de juego a partir de la posición del balón.

## Apariencia y respuesta

En `juego_3d.py`, `ESCENARIO_3D = "estadio"` activa gradas, publico y
reflectores. Puedes cambiarlo a `"barrio"` para usar la variante comunitaria
con casas y malla. `ASSET_STYLE = "3d"` selecciona `assets_3d`.

Los jugadores 2D tienen silueta orientada, uniforme, extremidades, cabeza y
sombra. La cancha 3D usa un encuadre de transmisión más alto, sprites de cuerpo
recortados por transparencia; ofrece el estadio y el entorno de barrio.
La vista táctica conserva los modelos 3D. En `juego_3d.py`, `ASSET_STYLE = "3d"`
selecciona `assets_3d`; puede cambiarse a `"2d"` para probar la otra carpeta.
La IA conserva su clasificador y su red entre reinicios, y la predicción Keras
se ejecuta directamente para reducir demora.

Para verificar modos, controles y física del juego 3D:

```powershell
Set-Location C:\OCT13\Futbol_IA_2D_3D
& C:\OCT13\.2d3d\Scripts\python.exe .\juego_3d.py --prueba-rapida
```
