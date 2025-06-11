
## 1. `AnalisisDescriptivo`

Clase para realizar análisis estadístico básico y estimación de densidad (histogramas y estimadores kernel).

### Métodos principales:

- `calculo_de_media()`: Devuelve la media muestral.
- `calculo_de_mediana()`: Devuelve la mediana.
- `calculo_varianza()`: Varianza muestral.
- `calculo_desvio_estandar()`: Desvío estándar.
- `calcular_cuartiles()`: Calcula los cuartiles (aunque falta el return).
- `calcular_histograma(h)`: Estima una función de densidad por histogramas.
- `evalua_histograma(h, x)`: Evalúa la densidad estimada por histogramas en una grilla `x`.
- `generar_datos_normal(media, desvio)`: Crea la densidad teórica normal sobre la grilla.
- `kernel_uniforme`, `kernel_gaussiano`, `kernel_cuadratico`, `kernel_triangular`: Diferentes núcleos para estimación.
- `mi_densidad(x, h, kernel)`: Estimación de densidad usando el kernel especificado.
- `qqplot()`: Grafica un QQ plot para verificar normalidad.

---

## 2. `GeneradoraDeDatos`

Clase para generar muestras aleatorias de diferentes distribuciones.

### Métodos principales:

- `generar_datos_dist_norm(media, desvio)`: Muestra normal.
- `pdf_norm(x, media, desvio)`: PDF teórica normal.
- `generar_datos_BS()`: Genera datos con mezcla de normales y ruido uniforme (tipo benchmark).
- `pdf_BS(x)`: Densidad teórica aproximada para los datos BS.
- `generar_datos_exp(beta)`: Datos exponenciales.
- `pdf_exp(x, beta)`: PDF de la exponencial.
- `generar_datos_chi2(gl)`: Datos chi-cuadrado.
- `pdf_chi2(x, gl)`: PDF chi-cuadrado.

---

## 3. `A_RegresionBase`

Clase abstracta base para regresiones (simple y múltiple).

### Métodos clave:

- `graficar()`: Gráfico de dispersión y recta ajustada.
- `resumen_modelo()`: Summary completo del modelo `statsmodels`.
- `estimador_varianza()`: Estima varianza del error.
- `intervalo_confianza_mu(x_0)`: Intervalo de confianza para media condicional.
- `intervalo_prediccion(x_0)`: Intervalo de predicción.
- `coeficiente_correlacion()`: Correlación X-Y.
- `r_cuadrado()`, `r_cuadrado_ajustado()`: R² y R² ajustado.
- `graficos_diagnostico()`: Residuales vs predichos y QQ plot.

---

## 4. `A_RegresionLinealSimple`

Hereda de `A_RegresionBase`, ajusta un modelo de regresión lineal simple a partir de un archivo `.csv`.

### Métodos propios:

- `test_hipotesis_beta1()`: Test sobre el coeficiente β₁.
- `intervalo_confianza_beta1()`: IC para β₁.
- `grafico_residuos()`: Residuos vs valores ajustados.
- `qqplot()`: QQ plot de residuos.

---

## 5. `A_RegresionLinealMultiple`

Hereda de `A_RegresionBase`, ajusta regresión lineal múltiple sobre un `DataFrame`.

### Métodos propios:

- `graficar_pares()`: Scatter plots de Y vs cada X.
- `coeficientes_correlacion()`: Correlaciones Y vs cada X.
- `vif()`: Calcula el VIF (Variance Inflation Factor) para cada variable → detecta multicolinealidad.
- `intervalo_confianza_coeficientes()`: ICs para todos los coeficientes.
- `predecir()`: Hace predicciones con nuevos datos.

---

## 6. `RegresionLogistica`

Clase para ajustar y evaluar un modelo de regresión logística.

### Flujo de uso:

1. `preparar_datos()`: Convierte variable respuesta a binaria y divide datos en train/test.
2. `ajustar_modelo()`: Ajusta modelo `Logit` usando `statsmodels`.
3. `obtener_resultados_modelo()`: Devuelve coeficientes, errores, p-valores.
4. `predecir(umbral=0.5)`: Predicción binaria.
5. `evaluar_modelo()`: Calcula matriz de confusión, sensibilidad, especificidad, error.
6. `curva_roc()`: Dibuja curva ROC y devuelve el área bajo la curva (AUC).

---

## 📦 Dependencias requeridas

```bash
pip install numpy pandas matplotlib scipy statsmodels scikit-learn