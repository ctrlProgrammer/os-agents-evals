# Roadmap

Qué viene para `os-agents-evals`, por prioridad. Primero velocidad, después que la muestra sea confiable y al final más métricas.

Reglas para todos los ítems (ver [README](README.md) para lo que ya existe):

- Un ítem es una historia: una rama `feature/<nombre>`, un ejemplo Given/When/Then escrito antes del código, test en rojo, mínimo código, verde.
- Los campos nuevos de `result/<versión>/<modelo>.json` se agregan, nunca se quitan ni se renombran. Los resultados viejos no los tienen: quien los lea debe tolerar un campo ausente.
- Cada ítem actualiza el README y no cambia lo que ya miden las corridas anteriores.

## Orden

| # | Ítem | Fase | Tamaño | Depende de |
|---|------|------|--------|------------|
| 1 | Llamadas en paralelo (`--workers`) | Velocidad | M | — |
| 2 | Muestreo estratificado en `--max-evals` | Muestra confiable | M | — |
| 3 | Intervalo de confianza de la exactitud | Muestra confiable | S | — |
| 4 | Matriz de confusión y precisión, recall y F1 por clase | Calidad | M | — |
| 5 | Tasa de respuestas inválidas | Calidad | S | — |
| 6 | Exactitud por umbral de confianza | Calidad | M | 1 (para tener casos suficientes) |
| 7 | Costo por acierto | Costo | S | — |
| 8 | Exactitud por grupo | Costo y agrupación | M | — |

## Fase 1. Velocidad

### 1. Llamadas en paralelo (`--workers N`)

**Por qué.** Hoy `run_eval` llama a cada caso en serie. 40 casos tardan cerca de 90 s en Gemini; el dataset `freezer` completo (6675) serían horas.

**Qué.** Flag `--workers N` (por defecto 1, igual que hoy). `run_eval` llama al modelo con un pool de hilos y conserva el orden de los casos. La latencia sigue medida por llamada, así que el p95 no cambia de significado. El resultado agrega `wall_s`, el tiempo total de la corrida, para ver la ganancia.

**Ejemplo.**
- Given 8 casos con un modelo de prueba que tarda 0,2 s por llamada.
- When corro con `--workers 4`.
- Then el tiempo total es menor que 0,6 s, el resultado es igual al de `--workers 1` y `per_label` conserva el orden.

**Fuera.** Reintentos ante 429 y 5xx. Con muchos hilos un 429 sigue tumbando la corrida; es una historia aparte y conviene hacerla justo después. `--workers` debe documentar ese riesgo.

## Fase 2. Muestra confiable

### 2. Muestreo estratificado en `--max-evals`

**Por qué.** El muestreo actual es uniforme: en 40 casos de `freezer` no salieron `hit`, `redbull` ni `congelador/sinmarca`, así que esas clases no se miden.

**Qué.** `--sample stratified` (nuevo valor por defecto) toma casi la misma cantidad por clase, con `--seed` (por defecto 7) para repetirlo. Una clase con menos casos que su cuota entrega todos los que tiene y el resto se reparte entre las demás. `--sample spread` conserva el muestreo actual. El resultado guarda `sample: {"strategy", "seed"}`, porque dos corridas con muestras distintas no son comparables.

**Ejemplo.**
- Given un dataset con 10 casos de A, 10 de B y 10 de C.
- When corro con `--max-evals 6`.
- Then hay 2 casos de cada clase y la misma `--seed` da los mismos 6 casos.

Otro ejemplo: con A de 50 casos, B de 50 y C de 1, `--max-evals 6` toma 1 de C y reparte los otros 5 entre A y B.

### 3. Intervalo de confianza de la exactitud

**Por qué.** Con 40 casos, un caso son 2,5 puntos. "95% contra 97,5%" no dice nada sin un margen.

**Qué.** Intervalo de Wilson al 95% de la exactitud. El resultado agrega `accuracy_ci95: [bajo, alto]` y la fila muestra `95% (IC 83-99%)`. Es una función pura de `(aciertos, casos)`.

**Ejemplo.**
- Given 38 aciertos de 40 casos.
- When calculo el intervalo.
- Then es aproximadamente `[0.835, 0.986]`.

**Fuera.** Comparar dos modelos con una prueba pareada (McNemar). Eso es de un futuro `compare`.

## Fase 3. Calidad

### 4. Matriz de confusión y precisión, recall y F1 por clase

**Por qué.** La exactitud global y los aciertos por etiqueta esconden qué se confunde con qué (por ejemplo `nevera/cocacola` contra `nevera/postobon`) y premian responder siempre la clase grande.

**Qué.** El resultado agrega `confusion: {esperada: {respondida: n}}`, con la respuesta que no es ninguna etiqueta conocida contada como `"(otra)"`. `per_label` suma `precision`, `recall` y `f1`, y mantiene `correct` y `total`. El resultado y la fila agregan el F1 macro. Todo se calcula en memoria con las respuestas de la corrida; no hace falta guardar caso por caso.

**Ejemplo.**
- Given esperadas `[a, a, b, b]` y respondidas `[a, b, b, b]`.
- When calculo las métricas.
- Then para `a` precisión 1,0 y recall 0,5; para `b` precisión 0,667 y recall 1,0; F1 macro 0,733.

### 5. Tasa de respuestas inválidas

**Por qué.** Gemini responde texto libre. Una respuesta como `Nevera/Cocacola.` o una frase larga hoy cuenta como fallo, igual que una etiqueta equivocada, y no sabemos cuál de las dos pasó.

**Qué.** El prompt (JSON de cualquier modelo) acepta una clave opcional `labels` con las respuestas válidas; en JEV, por defecto son las claves de `criteria`. El resultado agrega `invalid_rate`, `invalid_count` y hasta 3 ejemplos recortados a 80 caracteres en `invalid_samples`. Sin `labels` el campo queda en `null`. Los datasets de respuesta abierta (MMMU) no la definen.

**Ejemplo.**
- Given `labels: ["a", "b"]` y un modelo que responde `"a"`, `"A. porque..."` y `"b"`.
- When corro el eval.
- Then `invalid_count` es 1, `invalid_rate` 0,333 y la exactitud sigue contando esa respuesta como fallo.

**Fuera.** Aceptar variantes (punto final, mayúsculas con acento). Esta métrica sirve justo para decidir si hace falta.

### 6. Exactitud por umbral de confianza

**Por qué.** JEV devuelve confianza, pero en la corrida `v1` su único fallo tuvo 0,99. Hay que medir si la confianza sirve para decidir qué mandar a revisión humana.

**Qué.** Para los modelos que reportan confianza, el resultado agrega `by_confidence`: para cada umbral `[0.5, 0.7, 0.9, 0.95, 0.99]`, la `coverage` (parte de los casos con confianza igual o mayor) y la `accuracy` dentro de ellos. Va solo en el JSON.

**Ejemplo.**
- Given cuatro casos con (confianza, acierto) = (1,0 sí), (0,9 sí), (0,6 no), (0,4 no).
- When calculo la tabla.
- Then con umbral 0,5 la cobertura es 0,75 y la exactitud 0,667; con 0,9 la cobertura es 0,5 y la exactitud 1,0.

**Nota.** Con pocos fallos la tabla es ruidosa; hace falta una corrida de cientos de casos, por eso depende de la historia 1.

## Fase 4. Costo y agrupación

### 7. Costo por acierto

**Por qué.** El costo por 1k consultas no compara un modelo barato que falla más contra uno caro que acierta más.

**Qué.** El resultado agrega `cost_per_1k_correct`: costo total entre aciertos, por mil. Queda en `null` si no hay aciertos. La fila agrega `$X/1k aciertos`.

**Ejemplo.**
- Given 2 casos que cuestan $0,00045 cada uno y 1 acierto.
- When calculo el costo por acierto.
- Then `cost_per_1k_correct` es 0,9.

### 8. Exactitud por grupo

**Por qué.** En `freezer` interesa saber si un modelo falla más en `nevera` que en `congelador`, o en una marca, sin leer la lista entera de etiquetas.

**Qué.** Un caso de `dataset.json` acepta un `meta` opcional, por ejemplo `{"tipo": "nevera", "marca": "cocacola"}`. El resultado agrega `per_group: {clave: {valor: {correct, total}}}`. Para `freezer` hay que regenerar `dataset.json` con ese campo (no está en git).

**Ejemplo.**
- Given tres casos con `meta.tipo` = nevera, nevera, congelador, de los que el primero y el tercero se acertaron.
- When corro el eval.
- Then `per_group.tipo` es `{"nevera": {"correct": 1, "total": 2}, "congelador": {"correct": 1, "total": 1}}`.

**Pregunta abierta.** Medir también el acierto solo del tipo cuando falla la marca (crédito parcial) cambiaría cómo se califica un caso. Queda fuera de esta historia.

## Más adelante

Ideas que salieron y no están priorizadas: guardar la respuesta de cada caso, reanudar una corrida cortada, caché de respuestas, reintentos ante 429, comando `compare`, más adaptadores (Claude, GPT, Ollama) y votación entre modelos.
