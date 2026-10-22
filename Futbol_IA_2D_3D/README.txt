PITCH / FÚTBOL IA 2D Y 3D
========================

Dos juegos independientes: uno 2D con Pygame y otro 3D con Ursina.
Cada juego incluye en su propio archivo la IA, el registro con Pandas, las selecciones, el audio y sus recursos visuales.

REQUISITOS
----------
- Windows 64 bits y Python 3.12 de 64 bits.
- Entorno virtual .2d3d activado antes de instalar o ejecutar.
- Instalar las dependencias desde esta carpeta:

    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt

La carpeta del entorno virtual no forma parte del ZIP: debe permanecer fuera
del código del juego. Si el proyecto está en C:\OCT13\Futbol_IA_2D_3D y el
entorno en C:\OCT13\.2d3d, activa el entorno desde PowerShell con:

    ..\.2d3d\Scripts\Activate.ps1

EJECUTAR
--------
Desde Futbol_IA_2D_3D, con .2d3d activo:

    python juego_2d.py
    python juego_3d.py

Solo esos dos archivos .py son necesarios para ejecutar los juegos; ya no dependen
de módulos Python auxiliares. El código anterior queda guardado en
respaldo_codigo_pre_fusion_20261014.zip.

El archivo principal juego_3d.py ahora inicia la versión 3D volumétrica. En
Visual Studio Code, abre este archivo en la raíz del proyecto y pulsa “Ejecutar
archivo de Python”; selecciona el intérprete .2d3d si VS Code lo solicita.

juego_3d_base.py conserva la lógica 3D anterior usada como base interna;
animador_procedural.py y estadio_procedural.py son módulos de apoyo. No ejecutes
juego_3d_base.py directamente.

MODOS
-----
1. Un usuario contra la IA.
2. Dos usuarios locales.
3. Una IA contra dos usuarios cooperativos.
4. Dos IA contra dos usuarios cooperativos.

CONTROLES
---------
En los dos juegos controlas al jugador activo y puedes cambiar entre los 11
jugadores, como en un juego de fútbol de consola. El portero se mueve con la IA
por defecto, pero también puedes seleccionarlo manualmente. Los compañeros no
seleccionados se mueven con la IA táctica. El cambio sigue automáticamente al
receptor de un pase y, al defender, al jugador más próximo al balón. En
cooperativo, cada usuario mantiene un jugador humano diferente.

Jugador 1: WASD para moverse, TAB para cambiar jugador, Espacio para tirar,
Q para pasar, E para centrar, F para disputar/recuperar y Shift izquierdo para correr.
Jugador 2: flechas para moverse, Retroceso (Backspace) para cambiar jugador,
Ctrl derecho para tirar, Enter para pasar, U para centrar, Alt derecho para
disputar/recuperar y Shift derecho para correr.
En el juego 3D, W y Flecha arriba avanzan hacia el fondo de la cancha;
S y Flecha abajo se acercan a la cámara. A y D / Flechas izquierda y derecha
se mueven hacia los lados.
La dirección del tiro sigue la orientación del jugador; se puede apuntar
mientras se mueve. El balón se conduce al tenerlo cerca, los pases buscan al
compañero (o van al espacio si no hay uno) y los jugadores de IA presionan y
ofrecen líneas de apoyo.
Después de cada gol, el equipo que lo recibió inicia desde el centro. En el
juego 3D, pulsa V para alternar entre cámara de transmisión y vista cenital 2D.
El menú 3D presenta los modos como una previa de partido y muestra una cancha
táctica en miniatura. La vista de transmisión acompaña suavemente la jugada;
el estadio 3D incluye uniformes con más detalle, tribunas techadas, marcador
físico e iluminación con sombras. Si el rendimiento baja, en juego_3d.py puedes cambiar
`shadows=True` por `shadows=False` en la creación de DirectionalLight.
P o Esc abre el menú de pausa, con botones para continuar, reiniciar y volver a
la selección. R reinicia el partido en cualquier momento. Esc desde la pausa
vuelve al menú principal; al terminar, el menú ofrece una revancha.

Los jugadores 2D muestran silueta orientada, uniforme, piernas, brazos, cabeza
y sombra. En 3D tienen proporciones humanoides low-poly, variedad de tonos de
piel, uniforme distinto para el portero, dorsal y animación de carrera. Para
reducir la demora del 3D, la IA conserva sus modelos al reiniciar, Keras usa
inferencia directa, y se optimizaron las sombras y la tribuna del estadio.

AUDIO Y NARRADOR
----------------
El ZIP incluye WAV para ambiente de estadio, gol, silbato y golpe al balón;
no es necesario buscar paquetes de sonidos. Se reproducen al iniciar los
juegos si Windows permite acceder al dispositivo de audio. El narrador anuncia
la posesión, pases, centros, tiros y goles mediante una voz local de Windows
(pyttsx3/SAPI; System.Speech se usa como respaldo). No necesita internet.
Si no se oye la voz, revisa assets/narrador_estado.txt después de iniciar una
partida y comprueba que Windows tenga una voz instalada y el volumen activo.

DATOS E IA
----------
Al iniciar un modo se ajusta una regresión logística de Scikit-learn y una red
Keras pequeña usando datos sintéticos de demostración. Los goles, tiros, pases
y resultados se guardan en CSV dentro de data/ al cerrar o finalizar el partido.
La primera carga puede tardar unos segundos mientras se inicializan los modelos.
