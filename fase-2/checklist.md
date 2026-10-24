# Checklist Fase 2: Scripts y Docker

**Entrega: 31 de octubre de 2026** · Vale el 15 % de la nota final.

Los grupos están ordenados en un orden lógico de trabajo, no en el del documento.

**Leyenda:** `[x]` hecho y verificado · `[ ] 🟡` código escrito, falta ejecutarlo o comprobarlo · `[ ]` pendiente

---

## 0. Preparación del repositorio y Git
- [x] Conservar la carpeta `fase-1` sin tocarla
- [x] Crear la carpeta `fase-2` en el mismo repositorio
- [ ] Crear la rama `develop` (si no existe) y trabajar con ramas `feature/*` (ej. `feature/train-script`, `feature/predict-script`, `feature/docker-tests`)
- [x] No hacer cambios de desarrollo directamente en `main`
- [x] Crear la estructura base:
  - [x] `train.py`, `predict.py`
  - [x] `src/` (lógica compartida, recomendado)
  - [x] `tests/test_aplicacion.py`
  - [x] `data/ejemplo_prediccion.csv`
  - [x] `models/`, `outputs/`
  - [x] `requirements.txt`, `Dockerfile`, `.dockerignore`, `.gitignore`, `README.md`

## 1. Correcciones de la Fase 1
- [ ] Revisar la retroalimentación de la Fase 1
- [ ] Aplicar las correcciones necesarias (preprocesamiento, validación, modelo)
  - Ya aplicado: partición agrupada por día (`GroupShuffleSplit` sobre `dteday`). Falta contrastar con la retroalimentación.
- [ ] 🟡 Documentar los cambios que afecten preprocesamiento, validación o modelo (ya está en el borrador del README, sección 2)
- [x] Verificar que la estrategia de evaluación sea consistente con el problema y sin fuga de información (días compartidos entre train y test = 0)
- [x] Mantener el mismo problema y dataset aprobados (no es obligatorio un algoritmo nuevo ni mejorar la métrica)

## 2. Organización del código
- [x] Separar en funciones/módulos: lectura y validación de datos, preprocesamiento, entrenamiento, evaluación y almacenamiento (`src/config.py`, `data.py`, `preprocessing.py`, `training.py`, `evaluation.py`, `storage.py`, `prediction.py`)
- [x] No copiar las celdas del notebook en un `.py`
- [x] Evitar duplicar transformaciones entre entrenamiento y predicción (un solo `build_preprocessor()` dentro del `Pipeline` guardado)
- [ ] 🟡 No entrenar ni leer archivos al importar un módulo (usar `if __name__ == "__main__":`) → comprobar con `python -c "import train"` y `python -c "import predict"`
- [ ] 🟡 Usar `argparse` para rutas y opciones
- [ ] 🟡 Que `--help` funcione en ambos scripts
- [x] Sin rutas absolutas, sin `input()` interactivo, sin depender de celdas previas

## 3. Preprocesamiento y reproducibilidad
- [x] Usar `Pipeline` + `ColumnTransformer` (o equivalente bien organizado)
- [ ] 🟡 Guardar el preprocesamiento junto con el modelo (misma versión, cargados juntos) → comprobar con `save_version` / `load_version`
- [ ] 🟡 Conservar nombres y orden de variables, reglas de nulos y codificaciones
- [x] No usar la variable objetivo, información posterior al evento ni datos de prueba para ajustar el preprocesamiento o seleccionar el modelo
- [ ] 🟡 Documentar la semilla y limitaciones si el algoritmo es no determinista (ya está en el borrador del README, sección 8)

## 4. `train.py`
**Argumentos mínimos**
- [ ] 🟡 Ruta del CSV de datos etiquetados
- [ ] 🟡 Nombre de la variable objetivo
- [ ] 🟡 Directorio de salida e identificador de versión
- [ ] 🟡 Semilla y parámetros de separación de datos

**Proceso**
- [ ] 🟡 Leer datos y verificar que existan la variable objetivo y las columnas necesarias
- [x] Separar train/test **antes** de ajustar transformaciones (respetar grupos naturales si aplica)
- [x] Ajustar imputación, codificación y escalamiento solo con train (o dentro de cada fold si hay validación cruzada)
- [x] Entrenar el modelo y evaluarlo con datos no usados para ajustarlo (R² test = 0,914)
- [x] Evaluar el modelo base con la misma partición y métrica de la Fase 1
- [ ] 🟡 Guardar modelo + preprocesamiento + información de la ejecución
- [ ] 🟡 Mostrar un resumen y las rutas de los archivos generados

**Salidas**
- [ ] 🟡 `modelo.joblib` (o equivalente)
- [ ] 🟡 `metrics.json` (modelo y modelo base)
- [ ] 🟡 `metadata.json`

## 5. `predict.py`
**Argumentos mínimos**
- [ ] 🟡 Ruta del modelo/versión
- [ ] 🟡 CSV de entrada (sin variable objetivo)
- [ ] 🟡 CSV de salida

**Validación y procesamiento**
- [ ] 🟡 Comprobar que el modelo y el CSV existan y se puedan leer
- [ ] 🟡 Validar columnas requeridas y tipos de datos compatibles
- [ ] 🟡 Ordenar variables según el esquema del entrenamiento (sin depender del orden del CSV)
- [ ] 🟡 Aplicar el tratamiento de nulos de la versión guardada (nulo ≠ columna ausente)
- [ ] 🟡 Definir y documentar qué pasa con columnas adicionales (se ignoran; documentado en el README)
- [ ] 🟡 Definir y documentar qué pasa con categorías no vistas en entrenamiento (se ignoran, one-hot en ceros; documentado en el README)
- [ ] 🟡 Una predicción por fila, conservando el orden; no eliminar filas en silencio
- [ ] 🟡 No reentrenar ni reajustar transformaciones

**Salida y errores**
- [ ] 🟡 Columna `prediccion` (o nombre equivalente documentado)
- [ ] 🟡 Relacionable con la fila de entrada (orden o identificador) → columna `fila` y `--id-col` opcional
- [x] Si es clasificación: explicar el significado de las etiquetas (no aplica: es regresión; `prediccion` = bicicletas estimadas por hora, documentado en el README)
- [ ] 🟡 Mensajes de error comprensibles (columna ausente, tipo incompatible, modelo inexistente)
- [ ] 🟡 Código de salida distinto de cero en errores
- [ ] 🟡 No generar un archivo de salida de apariencia válida si el proceso falla (escritura vía archivo `.tmp`)

## 6. Versionamiento del modelo
- [ ] 🟡 Estructura `models/v1/`, `models/v2/` con modelo, métricas y metadatos
- [ ] 🟡 No sobrescribir una versión existente (rechazar el ID repetido o generar uno nuevo)
- [ ] 🟡 Poder elegir la versión explícitamente en `predict.py`
- [ ] **Generar al menos 2 versiones** con `train.py` y demostrar que ambas se cargan (hacer commit antes; sugerencia: `v2` con `--min-samples-leaf 5`)
- [ ] Dejar los artefactos accesibles en el repo, o con enlace de descarga documentado si son muy grandes (revisar el tamaño de los `.joblib`; GitHub rechaza archivos de más de 100 MB)

**Metadatos mínimos en `metadata.json`:**
- [ ] 🟡 ID de versión y fecha de creación
- [ ] 🟡 Tipo de problema, variable objetivo y algoritmo
- [ ] 🟡 Nombres de variables de entrada y tipos esperados
- [ ] 🟡 Parámetros principales del modelo y semilla
- [ ] 🟡 Datos usados: fuente, archivo/versión, número de observaciones
- [ ] 🟡 Estrategia de separación, tamaños de train/test y métrica
- [ ] 🟡 Versión de Python y referencia al archivo de dependencias
- [ ] 🟡 Hash del commit del código usado

## 7. Dependencias y datos
- [ ] `requirements.txt` con versiones explícitas, incluyendo `pytest`, sin paquetes innecesarios
- [ ] Indicar la versión de Python en el README (misma en local y Docker)
- [ ] 🟡 Documentar cómo obtener los datos, qué archivos se requieren y dónde ubicarlos (en el borrador del README, sección 5)
- [ ] `prepare_data.py` (inyección MCAR de nulos, semilla 42, 1,2 % en `hum` y 1,7 % en `windspeed`), documentado en el README y ejecutable también en Docker
- [ ] El entrenamiento no debe depender de Kaggle/Colab ni de un notebook abierto
- [ ] Sin credenciales en el repo ni en la imagen Docker
- [ ] `.gitignore` que excluya entornos virtuales, cachés, temporales, y datos/artefactos que no deban ir en Git
- [ ] 🟡 `data/ejemplo_prediccion.csv`: CSV pequeño, con el esquema real y sin la variable objetivo (incluir alguna fila con `hum` o `windspeed` vacíos)

## 8. Docker
- [ ] `Dockerfile` con imagen base de Python de versión explícita (nada de `latest`)
- [ ] Definir `WORKDIR` (ej. `/app`), instalar dependencias, copiar el código
- [ ] Sin `ENTRYPOINT` que impida ejecutar `train.py` y `predict.py` con `docker run`
- [ ] **No** entrenar durante `docker build`
- [ ] `.dockerignore` que excluya venv, cachés, `.git`, credenciales y archivos innecesarios
- [ ] Datos montados desde el host con `-v`
- [ ] Modelos, métricas, metadatos y predicciones persisten fuera del contenedor (volúmenes)
- [ ] Pasar `GIT_COMMIT` al contenedor para que el hash quede en `metadata.json` (`-e GIT_COMMIT=...`)
- [ ] Verificar manualmente: `docker build`, entrenamiento en contenedor y predicción en contenedor
- [ ] Documentar los comandos reales e indicar la terminal usada (adaptar sintaxis si es PowerShell)

## 9. Pruebas con pytest (mínimo 5)
Se ejecutan con `python -m pytest` desde `fase-2`.
- [ ] **Entrenamiento y almacenamiento:** con datos pequeños se genera un artefacto cargable, métricas y metadatos
- [ ] **Carga y predicción:** una predicción por fila y resultados iguales a los del modelo antes de guardar (con tolerancia numérica)
- [ ] **Valores faltantes:** entrada válida con nulos se procesa según la estrategia documentada
- [ ] **Entrada inválida:** columna ausente produce error claro (y código de salida ≠ 0 si se prueba por CLI)
- [ ] **Versionamiento:** una segunda versión conserva la primera, ambas se cargan, y un ID repetido no sobrescribe
- [ ] Extra recomendada: la partición agrupada no comparte días entre train y test

**Condiciones:**
- [ ] Datos pequeños o sintéticos, sin descargas de red
- [ ] Usar `tmp_path` para no tocar los modelos de la entrega
- [ ] Usar `assert` / `pytest.raises` (imprimir no cuenta)
- [ ] No reentrenar con todo el dataset; deben correr en un computador personal

## 10. README
Borrador disponible en `README.md` (secciones 🚧 por completar).
- [ ] 🟡 Título, integrantes, problema y relación con la Fase 1
- [ ] 🟡 Correcciones/cambios respecto a la entrega anterior
- [ ] Versión de Python, creación del entorno e instalación de dependencias
- [ ] 🟡 Fuente de los datos, ubicación esperada y esquema del CSV de predicción
- [ ] 🟡 Comandos completos de entrenamiento, predicción y `--help` (falta ejecutarlos para confirmarlos)
- [ ] 🟡 Ubicación de versiones, métricas, metadatos y resultados
- [ ] Comandos de Docker (build y run) con montajes
- [ ] Comando de pytest y breve descripción de las pruebas
- [ ] 🟡 Métrica obtenida, comparación con el modelo base y limitaciones conocidas (valores provisionales; actualizar con `v1`)
- [ ] Fuentes consultadas y código externo adaptado
- [ ] Evidencia breve de ejecución
- [ ] Commit o etiqueta de Git de la Entrega 2 (ej. `entrega-2`)

## 11. Trabajo colaborativo
- [ ] Cada integrante hace contribuciones sustantivas y verificables
- [ ] Cada integrante crea al menos un Pull Request hacia `develop`
- [ ] Cada PR es revisado por otro integrante antes del merge
- [ ] Commits descriptivos (no cambios superficiales de último momento)
- [ ] Al final, integrar `develop` en `main`
- [ ] Crear la etiqueta `entrega-2` (recomendado)

## 12. Verificación final y entrega
Simula la revisión del profesor desde una copia nueva del repo:
- [ ] Clonar el repo, preparar datos y dependencias según el README
- [ ] Correr `python -m pytest` y que todo pase
- [ ] Entrenar una versión nueva y revisar sus artefactos
- [ ] Predecir con el archivo de ejemplo
- [ ] Construir la imagen Docker y ejecutar entrenamiento y predicción
- [ ] Confirmar que los resultados persisten fuera del contenedor
- [ ] Probar una entrada inválida y confirmar el error comprensible
- [ ] Diligenciar el **formulario de Entrega 2**

---

## Siguientes pasos sugeridos
1. Ejecutar `train.py` y `predict.py` con las pruebas manuales (`--help`, versión repetida, columna ausente, tipo incompatible) para convertir los 🟡 en `[x]`.
2. `prepare_data.py`.
3. `requirements.txt` con versiones fijas (y fijar la versión de Python).
4. Hacer commit y generar las versiones `v1` y `v2` definitivas.
5. Pruebas con pytest.
6. Docker y README final.

## Peso en la evaluación (para priorizar)
| Criterio | Peso |
|---|---|
| Funcionamiento (CLI, versiones, Docker, persistencia) | 30 % |
| Calidad técnica (código reutilizable, sin fuga, errores, versionamiento) | 25 % |
| Pruebas | 15 % |
| Documentación | 15 % |
| Entregables | 10 % |
| Trabajo colaborativo | 5 % |