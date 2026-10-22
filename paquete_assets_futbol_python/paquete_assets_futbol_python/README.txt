PAQUETE DE RECURSOS PARA VIDEOJUEGO DE FUTBOL EN PYTHON / PYGAME
=================================================================

CONTENIDO
---------
assets_2d/  Recursos con estilo 2D ilustrado, adecuados para un juego 2D.
assets_3d/  Recursos renderizados con aspecto volumetrico/realista. Son imagenes 2D prerenderizadas; no son modelos 3D ni archivos .obj/.fbx.

Cada carpeta contiene categorias:
  estadios/             Fondos panoramicos de estadio (tres variantes).
  cancha/               Cancha completa y una lamina de texturas/lineas.
  porterias/            Porterias frontal y lateral.
  jugadores/             Tiras de sprites de jugadores, equipos y porteros/arbitros.
  hinchas/               Tiras de hinchas celebrando.
  banderas/              Banderas/trapos de aficionados.
  balon_y_efectos/       Balones y efectos/celebraciones.

IMPORTANTE SOBRE LOS SPRITES
----------------------------
Las tiras proceden de una ilustracion generada y recortada por categorias. Algunas imagenes de sprites tienen transparencias aproximadas eliminando el fondo cuadriculado; revisa bordes y recorta cada personaje si necesitas una animacion con un frame por archivo. Las tiras no son un atlas con metadatos de coordenadas ni una animacion ya configurada.

INTEGRACION SEGURA CON UN PROYECTO EXISTENTE
--------------------------------------------
1. Descomprime este ZIP dentro de tu proyecto, sin reemplazar archivos existentes.
2. Conserva el codigo del juego actual. No copies archivos Python por encima de los tuyos sin revisar los cambios.
3. Abre Codex en la carpeta raiz del proyecto y pega el contenido de CODEX_INTEGRATION_PROMPT.txt.
4. Pide que detecte la resolucion y el sistema de coordenadas que ya usa el juego; no debe reescribirlo desde cero.
5. Antes de usar un recurso, cargalo desde una ruta relativa a pathlib.Path(__file__).resolve().parent o a la raiz del proyecto.
6. Usa pygame.image.load(ruta).convert_alpha() en PNG con transparencia. Usa convert() para fondos opacos.
7. Escala los fondos de estadio al area de pantalla y la cancha al area de juego. Mantiene la relacion de aspecto siempre que sea posible.
8. Para animar un jugador, separa la tira en frames individuales o crea un sprite sheet con coordenadas explicitamente verificadas.
9. Incluye fallback: si un archivo no existe o falla, muestra el recurso grafico actual o un rectangulo temporal, sin cerrar el juego.
10. Ejecuta y prueba el juego despues de cada cambio. Evita modificar logica de movimiento, colisiones, marcador o controles que ya funcione.

LICENCIA / USO
--------------
Recursos visuales generados para servir de base al proyecto. Revisa el acabado visual y adapta los recortes a las necesidades del juego antes de distribuirlo.
