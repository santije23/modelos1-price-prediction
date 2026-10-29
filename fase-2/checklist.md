# Checklist Fase 2: Scripts y Docker


**Leyenda:** `[x]` hecho y verificado · `[ ] 🟡` código escrito o evidencia parcial, falta ejecutarlo o comprobarlo · `[ ]` pendiente

---

## 0. Preparación del repositorio y Git
- [x] Conservar la carpeta `fase-1` sin tocarla
- [x] Crear la carpeta `fase-2` en el mismo repositorio
- [x] Crear la rama `develop` (si no existe) y trabajar con ramas `feature/*` (ej. `feature/train-script`, `feature/predict-script`, `feature/docker-tests`)
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
- [x] Documentar los cambios que afecten preprocesamiento, validación o modelo (README, sección 2)
- [x] Verificar que la estrategia de evaluación sea consistente con el problema y sin fuga de información (días compartidos entre train y test = 0)
- [x] Mantener el mismo problema y dataset aprobados (no es obligatorio un algoritmo nuevo ni mejorar la métrica)

## 2. Organización del código
- [x] Separar en funciones/módulos: lectura y validación de datos, preprocesamiento, entrenamiento, evaluación y almacenamiento (`src/config.py`, `data.py`, `preprocessing.py`, `training.py`, `evaluation.py`, `storage.py`, `prediction.py`)
- [x] No copiar las celdas del notebook en un `.py`
- [x] Evitar duplicar transformaciones entre entrenamiento y predicción (un solo `build_preprocessor()` dentro del `Pipeline` guardado)
- [x] No entrenar ni leer archivos al importar un módulo (usar `if __name__ == "__main__":`) → comprobado con `python -c "import train"` y `python -c "import predict"` (`pruebas.md`)
- [x] Usar `argparse` para rutas y opciones
- [x] Que `--help` funcione en ambos scripts
- [x] Sin rutas absolutas, sin `input()` interactivo, sin depender de celdas previas

## 3. Preprocesamiento y reproducibilidad
- [x] Usar `Pipeline` + `ColumnTransformer` (o equivalente bien organizado)
- [ ] 🟡 Guardar el preprocesamiento junto con el modelo (misma versión, cargados juntos) → falta la comprobación explícita (`verificar_pipeline.py`: pasos del pipeline, preprocesador ajustado y predicción sobre datos crudos con nulos)
- [x] Conservar nombres y orden de variables, reglas de nulos y codificaciones (nombres y tipos en `metadata.json`)
- [x] No usar la variable objetivo, información posterior al evento ni datos de prueba para ajustar el preprocesamiento o seleccionar el modelo
- [x] Documentar la semilla y limitaciones si el algoritmo es no determinista (README, sección 8)

## 4. `train.py`
**Argumentos mínimos**
- [x] Ruta del CSV de datos etiquetados
- [x] Nombre de la variable objetivo
- [x] Directorio de salida e identificador de versión
- [x] Semilla y parámetros de separación de datos

**Proceso**
- [x] Leer datos y verificar que existan la variable objetivo y las columnas necesarias (archivo inexistente y objetivo inexistente probados)
- [x] Separar train/test **antes** de ajustar transformaciones (respetar grupos naturales si aplica)
- [x] Ajustar imputación, codificación y escalamiento solo con train (o dentro de cada fold si hay validación cruzada)
- [x] Entrenar el modelo y evaluarlo con datos no usados para ajustarlo (R² test = 0,914)
- [x] Evaluar el modelo base con la misma partición y métrica de la Fase 1
- [x] Guardar modelo + preprocesamiento + información de la ejecución
- [x] Mostrar un resumen y las rutas de los archivos generados

**Salidas**
- [x] `modelo.joblib` (o equivalente)
- [x] `metrics.json` (modelo y modelo base)
- [x] `metadata.json`

## 5. `predict.py`
**Argumentos mínimos**
- [x] Ruta del modelo/versión
- [x] CSV de entrada (sin variable objetivo)
- [x] CSV de salida

**Validación y procesamiento**
- [x] Comprobar que el modelo y el CSV existan y se puedan leer
- [x] Validar columnas requeridas y tipos de datos compatibles
- [ ] 🟡 Ordenar variables según el esquema del entrenamiento (sin depender del orden del CSV) → falta probar con un CSV con las columnas en orden invertido
- [ ] 🟡 Aplicar el tratamiento de nulos de la versión guardada (nulo ≠ columna ausente) → comprobar con `ejemplo_prediccion.csv` (el resumen debe decir "valores nulos imputados: 2"); si las capturas 12/13 lo muestran, ya se puede marcar
- [x] Definir y documentar qué pasa con columnas adicionales (se ignoran; README, sección 4)
- [x] Definir y documentar qué pasa con categorías no vistas en entrenamiento (se ignoran, one-hot en ceros; README, sección 4)
- [x] Una predicción por fila, conservando el orden; no eliminar filas en silencio
- [x] No reentrenar ni reajustar transformaciones

**Salida y errores**
- [x] Columna `prediccion` (o nombre equivalente documentado)
- [x] Relacionable con la fila de entrada (orden o identificador) → columna `fila` y `--id-col` opcional
- [x] Si es clasificación: explicar el significado de las etiquetas (no aplica: es regresión; `prediccion` = bicicletas estimadas por hora, documentado en el README)
- [x] Mensajes de error comprensibles (modelo inexistente, columna ausente y tipo incompatible probados)
- [x] Código de salida distinto de cero en errores
- [x] No generar un archivo de salida de apariencia válida si el proceso falla (escritura vía archivo `.tmp`)

## 6. Versionamiento del modelo
- [x] Estructura `models/v1/`, `models/v2/` con modelo, métricas y metadatos
- [x] No sobrescribir una versión existente (rechazar el ID repetido o generar uno nuevo) → comprobado con `train.py`; la protección interna de `save_version` se cubrirá además con pytest
- [ ] 🟡 Poder elegir la versión explícitamente en `predict.py` → se cierra al tener `v1` y `v2` y predecir con ambas
- [ ] **Generar al menos 2 versiones** con `train.py` y demostrar que ambas se cargan (hacer commit antes; sugerencia: `v2` con `--min-samples-leaf 5`)
- [ ] Dejar los artefactos accesibles en el repo, o con enlace de descarga documentado si son muy grandes (revisar el tamaño de los `.joblib`; GitHub rechaza archivos de más de 100 MB)

**Metadatos mínimos en `metadata.json`:**
- [x] ID de versión y fecha de creación
- [x] Tipo de problema, variable objetivo y algoritmo
- [x] Nombres de variables de entrada y tipos esperados
- [x] Parámetros principales del modelo y semilla
- [x] Datos usados: fuente, archivo/versión, número de observaciones
- [x] Estrategia de separación, tamaños de train/test y métrica
- [x] Versión de Python y referencia al archivo de dependencias
- [x] Hash del commit del código usado

## 7. Dependencias y datos
- [ ] `requirements.txt` con versiones explícitas, incluyendo `pytest`, sin paquetes innecesarios
- [ ] Indicar la versión de Python en el README (misma en local y Docker)
- [x] Documentar cómo obtener los datos, qué archivos se requieren y dónde ubicarlos (README, sección 5)
- [x] ~~`prepare_data.py`~~ Descartado: se usa el CSV generado en la Fase 1 (`data/processed/bike_share_with_nans.csv`) y su origen está documentado en el README
- [ ] 🟡 `data/processed/bike_share_with_nans.csv` incluido en Git (verificar que `.gitignore` no lo excluya) para que quien evalúe pueda entrenar sin pasos extra
- [ ] El entrenamiento no debe depender de Kaggle/Colab ni de un notebook abierto (se cumple si el CSV procesado está en el repo)
- [ ] Sin credenciales en el repo ni en la imagen Docker
- [ ] `.gitignore` que excluya entornos virtuales, cachés, temporales, y datos/artefactos que no deban ir en Git
- [ ] 🟡 `data/ejemplo_prediccion.csv`: CSV pequeño, con el esquema real y sin la variable objetivo (incluir alguna fila con `hum` o `windspeed` vacíos) → verificar que existe, que tiene los nulos y que está en Git

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
Se ejecutan con `python -m pytest` desde `fase-2`. (Las pruebas **manuales** de ambos scripts ya están documentadas en `pruebas.md` y en el README, sección 10.1; no reemplazan a pytest.)
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
- [x] Título, integrantes, problema y relación con la Fase 1
- [x] Correcciones/cambios respecto a la entrega anterior
- [ ] Versión de Python, creación del entorno e instalación de dependencias
- [x] Fuente de los datos, ubicación esperada y esquema del CSV de predicción
- [x] Comandos completos de entrenamiento, predicción y `--help` (ejecutados en las pruebas manuales)
- [ ] 🟡 Ubicación de versiones, métricas, metadatos y resultados (estructura documentada; falta describir `v1` y `v2` entregadas)
- [ ] Comandos de Docker (build y run) con montajes
- [ ] Comando de pytest y breve descripción de las pruebas (la parte de pruebas manuales ya está, sección 10.1)
- [ ] 🟡 Métrica obtenida, comparación con el modelo base y limitaciones conocidas (valores provisionales; actualizar con `v1`)
- [ ] Fuentes consultadas y código externo adaptado
- [ ] 🟡 Evidencia breve de ejecución (pruebas manuales de `train.py` y `predict.py` incluidas; falta la de Docker y pytest)
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
1. Cerrar los 🟡 de código que faltan: orden de columnas y nulos en `predict.py`, y `verificar_pipeline.py` (preprocesamiento guardado con el modelo).
2. Confirmar que `bike_share_with_nans.csv` y `ejemplo_prediccion.csv` están en Git (revisar `.gitignore`).
3. `requirements.txt` con versiones fijas (y fijar la versión de Python en el README).
4. Hacer commit y generar las versiones `v1` y `v2` definitivas; actualizar métricas del README.
5. Pruebas con pytest.
6. Docker y README final (con commit o etiqueta de la entrega).

## Peso en la evaluación (para priorizar)
| Criterio | Peso |
|---|---|
| Funcionamiento (CLI, versiones, Docker, persistencia) | 30 % |
| Calidad técnica (código reutilizable, sin fuga, errores, versionamiento) | 25 % |
| Pruebas | 15 % |
| Documentación | 15 % |
| Entregables | 10 % |
| Trabajo colaborativo | 5 % |
