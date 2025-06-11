import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, expon, chi2
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

class AnalisisDescriptivo:
  def __init__(self, datos):
        self.datos = np.array(datos)

  def calculo_de_media(self):
    return np.mean(self.datos)

  def calculo_de_mediana(self):
    return np.median(self.datos)

  def calculo_varianza(self):
    return np.var(self.datos)

  def calculo_desvio_estandar(self):
    return np.std(self.datos)

  def calcular_cuartiles(self):
    np.percentile(self.datos, [25, 50, 75])

  def calcular_histograma(self,h):
    puntos=np.arange(self.minimo(),self.maximo()+h,h)
    histograma=np.zeros(len(puntos)-1)
    for i in range(len(puntos)-1):
      for j in range(len(self.datos)):
        if self.datos[j]>=puntos[i] and self.datos[j]<puntos[i+1]:
          histograma[i]+=1
      histograma[i]=histograma[i]/len(self.datos)
      histograma[i]=histograma[i]/h
    return histograma,puntos

  def evalua_histograma(self,h,x):
    histograma,puntos=self.calcular_histograma(h)
    estim_hist=np.zeros(len(x))
    for i in range(len(x)):
      for j in range(len(puntos)-1):
        if x[i]>=puntos[j] and x[i]<puntos[j+1]:
          estim_hist[i]=histograma[j]
    return estim_hist

  def generar_datos_normal(self,media,desvio,N=1000):
    max=self.maximo()
    min=self.minimo()
    grilla=np.linspace(min,max,N)
    self.datos_normal=norm.pdf(grilla,media,desvio)
    return grilla,self.datos_normal

  def generar_datos_BS(self,N=1000):
    u = np.random.uniform(size=(n,))
    y = u.copy()
    ind = np.where(u > 0.5)[0]
    y[ind] = np.random.normal(0, 1, size=len(ind))
    for j in range(5):
        ind = np.where((u > j * 0.1) & (u <= (j+1) * 0.1))[0]
        y[ind] = np.random.normal(j/2 - 1, 1/10, size=len(ind))
    return y

  def kernel_uniforme(self,x):
    if (x>=-1/2) and (x<=1/2):
      valor_kernel_uniforme=1
    else:
      valor_kernel_uniforme=0
    return valor_kernel_uniforme

  def kernel_gaussiano(self,x):
    valor_kernel_gaussiano=1/np.sqrt(2*np.pi)*np.exp(-x**2/2)
    return valor_kernel_gaussiano

  def kernel_cuadratico(self,x):
    if (x>=-1/2) and (x<=1/2):
      valor_kernel_cuadratico=(3/4)*(1-x**2)
    else:
      valor_kernel_cuadratico=0
    return valor_kernel_cuadratico

  def kernel_triangular(self,x):
    valor_kernel_triangular=(1+x)*(x>=0 and x<=1)+(1-x)*(x<0 and x>-1)
    return valor_kernel_triangular

  def mi_densidad(self,x,h,kernel):
    if kernel=="uniforme":
      density=np.zeros_like(x,dtype=float)
      for i in range(len(x)):
        for j in range(len(self.datos)):
          density[i]+=self.kernel_uniforme((self.datos[j]-x[i])/h)
      density=density/(len(self.datos)*h)
    elif kernel=="gaussiano":
      density=np.zeros_like(x,dtype=float)
      for i in range(len(x)):
        for j in range(len(self.datos)):
          density[i]+=self.kernel_gaussiano((self.datos[j]-x[i])/h)
      density=density/(len(self.datos)*h)
    elif kernel=="cuadratico":
      density=np.zeros_like(x,dtype=float)
      for i in range(len(x)):
        for j in range(len(self.datos)):
          density[i]+=self.kernel_cuadratico((self.datos[j]-x[i])/h)
      density=density/(len(self.datos)*h)
    elif kernel=="triangular":
      density=np.zeros_like(x,dtype=float)
      for i in range(len(x)):
        for j in range(len(self.datos)):
          density[i]+=self.kernel_triangular((self.datos[j]-x[i])/h)
      density=density/(len(self.datos)*h)
    return density

  def qqplot(self, data=None):
    if data is None:
      data = self.datos
    x_ord = np.sort(data)
    n = len(x_ord)
    cuantiles_muestrales = [(i + 1 - 0.5) / n for i in range(n)]
    cuantiles_teoricos = norm.ppf(cuantiles_muestrales)
    plt.scatter(cuantiles_teoricos, x_ord, color='blue', marker='o')
    plt.xlabel('Cuantiles teóricos')
    plt.ylabel('Cuantiles muestrales')
    plt.plot(cuantiles_teoricos, cuantiles_teoricos, linestyle='-', color='red')
    plt.show()



class GeneradoraDeDatos:
    def __init__(self, n):
        """Clase para generar datos con distribuciones conocidas."""
        self.n = n

    def generar_datos_dist_norm(self, media, desvio):
        return np.random.normal(media, desvio, self.n)

    def pdf_norm(self, x, media, desvio):
        return norm.pdf(x, media, desvio)

    def generar_datos_BS(self):
        u = np.random.uniform(size=self.n)
        y = u.copy()
        ind = np.where(u > 0.5)[0]
        y[ind] = np.random.normal(0, 1, size=len(ind))
        for j in range(5):
            ind = np.where((u > j*0.1) & (u <= (j+1)*0.1))[0]
            y[ind] = np.random.normal(j/2 - 1, 1/10, size=len(ind))
        return y

    def pdf_BS(self, x):
        term1 = (1/2) * norm.pdf(x, 0, 1)
        term2 = (1/10) * sum(norm.pdf(x, j/2 - 1, 1/10) for j in range(5))
        return term1 + term2

    def generar_datos_exp(self, beta):
        return np.random.exponential(beta, self.n)

    def pdf_exp(self, x, beta):
        return expon.pdf(x, scale=beta)

    def generar_datos_chi2(self, gl):
        return np.random.chisquare(gl, self.n)

    def pdf_chi2(self, x, gl):
        return chi2.pdf(x, gl)




class A_RegresionBase:
    """Clase base con métodos comunes conservando toda la funcionalidad original"""

    def __init__(self):
        self.modelo = None
        self.resultados = None

    def _ajustar_modelo(self):
        """Método para ajustar el modelo (implementado en clases hijas)"""
        raise NotImplementedError

    def residuos(self):
        """Calcula los residuos del modelo."""
        return self.respuesta - self.valores_recta()

    def valores_recta(self, x=None):
        """Calcula los valores predichos (implementado en clases hijas)"""
        raise NotImplementedError

    def graficar(self, titulo="Regresión"):
        """Genera gráfico de dispersión con la recta de regresión."""
        plt.scatter(self.predictora, self.respuesta, marker="o", facecolors="none", edgecolors="blue")
        plt.plot(self.predictora, self.valores_recta(), color="red")
        plt.xlabel(self.nombre_predictora)
        plt.ylabel(self.nombre_respuesta)
        plt.title(titulo)
        plt.show()

    def resumen_modelo(self):
        """Muestra un resumen completo del modelo de regresión."""
        print(self.modelo.summary())

    def estimador_varianza(self):
        """Calcula la estimación de la varianza del error."""
        residuos = self.residuos()
        return np.sum(residuos**2) / (len(residuos) - 2)

    def intervalo_confianza_mu(self, x_0, confianza=0.95):
        """Calcula intervalo de confianza para la media condicional E(Y|X=x_0)."""
        alpha = 1 - confianza
        y_hat = self.b0 + self.b1 * x_0
        t_val = stats.t.ppf(1 - alpha/2, len(self.predictora)-2)
        margen_error = t_val * self._se_confianza(x_0)

        ic_inf = y_hat - margen_error
        ic_sup = y_hat + margen_error
        return (ic_inf, ic_sup)

    def intervalo_prediccion(self, x_0, confianza=0.95):
        """Calcula intervalo de predicción para un valor individual Y."""
        alpha = 1 - confianza
        y_hat = self.b0 + self.b1 * x_0
        t_val = stats.t.ppf(1 - alpha/2, len(self.predictora)-2)
        margen_error = t_val * self._se_prediccion(x_0)

        ic_inf = y_hat - margen_error
        ic_sup = y_hat + margen_error
        return (ic_inf, ic_sup)

    def _se_confianza(self, x_0):
        """Calcula el error estándar para el intervalo de confianza de la media."""
        x = self.predictora
        sigma2_est = self.estimador_varianza()

        sum_cuadrados = np.sum((x - np.mean(x))**2)
        var_beta1 = sigma2_est / sum_cuadrados
        var_beta0 = sigma2_est * np.sum(x**2) / (len(x) * sum_cuadrados)
        cov_01 = -np.mean(x) * sigma2_est / sum_cuadrados

        se2_est = var_beta0 + (x_0**2) * var_beta1 + 2 * x_0 * cov_01
        return np.sqrt(se2_est)

    def _se_prediccion(self, x_0):
        """Calcula el error estándar para el intervalo de predicción."""
        return np.sqrt(self._se_confianza(x_0)**2 + self.estimador_varianza())

    def predecir(self, x_0):
        """Predice el valor de Y para un valor dado de X."""
        return self.b0 + self.b1 * x_0

    def coeficiente_correlacion(self):
        """Calcula el coeficiente de correlación entre X e Y."""
        return np.corrcoef(self.predictora, self.respuesta)[0, 1]

    def r_cuadrado(self):
        """Retorna el coeficiente de determinación R²."""
        if hasattr(self.modelo, 'rsquared'):
            return self.modelo.rsquared
        return None

    def r_cuadrado_ajustado(self):
        """Retorna el R² ajustado."""
        if hasattr(self.modelo, 'rsquared_adj'):
            return self.modelo.rsquared_adj
        return None

    def graficos_diagnostico(self):
        """Genera gráficos de diagnóstico para evaluar supuestos."""
        residuos = self.residuos()
        predichos = self.valores_recta()

        plt.figure(figsize=(12, 5))

        # Gráfico de residuos vs predichos
        plt.subplot(1, 2, 1)
        plt.scatter(predichos, residuos, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Valores predichos')
        plt.ylabel('Residuos')
        plt.title('Residuos vs. Valores predichos')

        # QQ Plot
        plt.subplot(1, 2, 2)
        sm.qqplot(residuos, line='s')
        plt.title('QQ Plot de residuos')

        plt.tight_layout()
        plt.show()

class A_RegresionLinealSimple(A_RegresionBase):
    def __init__(self, datos_csv, col_predictora='X', col_respuesta='Y', sep=','):
        super().__init__()
        self.datos = pd.read_csv(datos_csv, sep=sep)
        self.predictora = self.datos[col_predictora].to_numpy()
        self.respuesta = self.datos[col_respuesta].to_numpy()
        self.nombre_predictora = col_predictora
        self.nombre_respuesta = col_respuesta
        self.b0, self.b1 = self._calcular_coeficientes(self.predictora, self.respuesta)
        self.modelo = self._ajustar_modelo()

    def _calcular_coeficientes(self, x, y):
        """Calcula los coeficientes de la recta de regresión."""
        media_x = np.mean(x)
        media_y = np.mean(y)
        b1 = np.sum((x - media_x) * (y - media_y)) / np.sum((x - media_x) ** 2)
        b0 = media_y - b1 * media_x
        return b0, b1

    def _ajustar_modelo(self):
        """Ajusta el modelo de regresión usando statsmodels."""
        X = sm.add_constant(self.predictora)
        modelo = sm.OLS(self.respuesta, X).fit()
        return modelo

    def valores_recta(self, x=None):
        """Calcula los valores predichos por la recta de regresión."""
        if x is None:
            x = self.predictora
        return self.b0 + self.b1 * np.asarray(x)

    def test_hipotesis_beta1(self, valor_h0=0, alpha=0.05):
        """Realiza test de hipótesis para el coeficiente beta1."""
        t_obs = (self.b1 - valor_h0) / self.modelo.bse[1]
        p_valor = 2 * (1 - stats.t.cdf(abs(t_obs), len(self.predictora)-2))

        print(f"Test para H0: β1 = {valor_h0} vs H1: β1 ≠ {valor_h0}")
        print(f"Estadístico t observado: {t_obs:.4f}")
        print(f"p-valor: {p_valor:.4f}")

        if p_valor < alpha:
            print(f"Conclusión: Rechazamos H0 (α={alpha}). Hay evidencia de que β1 ≠ {valor_h0}")
        else:
            print(f"Conclusión: No rechazamos H0 (α={alpha}). No hay evidencia suficiente para afirmar que β1 ≠ {valor_h0}")

    def intervalo_confianza_beta1(self, confianza=0.95):
        """Calcula intervalo de confianza para el coeficiente beta1."""
        alpha = 1 - confianza
        t_val = stats.t.ppf(1 - alpha/2, len(self.predictora)-2)
        margen_error = t_val * self.modelo.bse[1]

        ic_inf = self.b1 - margen_error
        ic_sup = self.b1 + margen_error

        print(f"Intervalo de confianza del {confianza*100:.0f}% para β1: ({ic_inf:.4f}, {ic_sup:.4f})")
        return (ic_inf, ic_sup)

    def grafico_residuos(self):
        """Genera gráfico de residuos vs valores ajustados."""
        predichos = self.valores_recta()
        residuos = self.residuos()

        plt.figure(figsize=(8, 5))
        plt.scatter(predichos, residuos, color='blue', alpha=0.6)
        plt.axhline(y=0, color='red', linestyle='--')
        plt.title('Residuos vs Valores Ajustados')
        plt.xlabel('Valores Ajustados')
        plt.ylabel('Residuos')
        plt.grid(True)
        plt.show()

    def qqplot(self):
        """Genera gráfico Q-Q para verificar normalidad de residuos."""
        residuos = self.residuos()
        plt.figure(figsize=(8, 5))
        sm.qqplot(residuos, line='s')
        plt.title('Gráfico Q-Q de Residuos')
        plt.show()

####################################                             ############################################                               ######################################                  #####################
class A_RegresionLinealMultiple(A_RegresionBase):
    def __init__(self, dataframe, variable_respuesta, variables_predictoras):
        super().__init__()
        self.df = dataframe.copy()
        self.respuesta = self.df[variable_respuesta].to_numpy()
        self.nombre_respuesta = variable_respuesta
        self.variables_predictoras = variables_predictoras
        self.X = sm.add_constant(self.df[variables_predictoras])
        self.modelo = sm.OLS(self.respuesta, self.X)
        self.resultados = self.modelo.fit()
        self.b0 = self.resultados.params[0]  # Intercepto
        self.predictora = self.df[variables_predictoras].to_numpy()  # Para compatibilidad con clase base

    def valores_recta(self):
        """Calcula los valores predichos por el modelo."""
        return self.resultados.fittedvalues

    def graficos_diagnostico(self):
        """Genera gráficos de diagnóstico para evaluar supuestos."""
        residuos = self.resultados.resid
        predichos = self.resultados.fittedvalues

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.scatter(predichos, residuos)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Valores predichos')
        plt.ylabel('Residuos')
        plt.title('Residuos vs. Valores predichos')

        plt.subplot(1, 2, 2)
        sm.qqplot(residuos, line='s')
        plt.title('QQ Plot de residuos')

        plt.tight_layout()
        plt.show()

    def graficar_pares(self):
        """Genera gráficos de dispersión para cada variable predictora vs la respuesta."""
        n_vars = len(self.variables_predictoras)
        fig, axes = plt.subplots(1, n_vars, figsize=(5*n_vars, 4))

        if n_vars == 1:
            axes = [axes]

        for i, var in enumerate(self.variables_predictoras):
            axes[i].scatter(self.df[var], self.respuesta, alpha=0.6)
            axes[i].set_xlabel(var)
            axes[i].set_ylabel(self.nombre_respuesta)
            axes[i].set_title(f'{var} vs {self.nombre_respuesta}')

        plt.tight_layout()
        plt.show()

    def coeficientes_correlacion(self):
        """Retorna un diccionario con los coeficientes de correlación entre cada predictora y la respuesta."""
        return {var: np.corrcoef(self.df[var], self.respuesta)[0, 1]
                for var in self.variables_predictoras}

    def predecir(self, nuevos_datos):
        """
        Predice valores para nuevos datos.

        Parámetros:
        nuevos_datos: DataFrame con las mismas columnas predictoras o array numpy 2D
        """
        if isinstance(nuevos_datos, pd.DataFrame):
            X_nuevo = sm.add_constant(nuevos_datos[self.variables_predictoras])
        else:
            if len(nuevos_datos.shape) == 1:
                nuevos_datos = nuevos_datos.reshape(1, -1)
            X_nuevo = sm.add_constant(nuevos_datos)

        return self.resultados.predict(X_nuevo)

    def vif(self):
        """Calcula el Factor de Inflación de Varianza para detectar multicolinealidad."""
        vif_data = pd.DataFrame()
        vif_data["Variable"] = self.variables_predictoras
        vif_data["VIF"] = [variance_inflation_factor(self.X.values, i+1)
                          for i in range(len(self.variables_predictoras))]
        return vif_data

    def intervalo_confianza_coeficientes(self, confianza=0.95):
        """Retorna intervalos de confianza para todos los coeficientes."""
        return self.resultados.conf_int(alpha=1-confianza)




class RegresionLogistica:
    def __init__(self, datos, variable_respuesta, test_size=0.3, random_state=1):
        self.datos = datos.copy()
        self.variable_respuesta = variable_respuesta
        self.test_size = test_size
        self.random_state = random_state
        self.modelo = None
        self.resultados = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        
    def preparar_datos(self):
        # Convertir variable respuesta a binario
        if self.datos[self.variable_respuesta].dtype == 'object':
            cats = self.datos[self.variable_respuesta].unique()
            self.datos[self.variable_respuesta] = (self.datos[self.variable_respuesta] == cats[0]).astype(int)
        
        # Dividir datos
        X = self.datos.drop(columns=[self.variable_respuesta])
        y = self.datos[self.variable_respuesta]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state)
        
        # Añadir constante
        self.X_train = sm.add_constant(self.X_train)
        self.X_test = sm.add_constant(self.X_test)
    
    def ajustar_modelo(self):
        self.modelo = sm.Logit(self.y_train, self.X_train)
        self.resultados = self.modelo.fit()
        print(self.resultados.summary())
        
    def obtener_resultados_modelo(self):
        return {
            'betas': self.resultados.params,
            'errores': self.resultados.bse,
            't_values': self.resultados.tvalues,
            'p_values': self.resultados.pvalues
        }
    
    def predecir(self, umbral=0.5, X_nuevo=None):
        if X_nuevo is None:
            X = self.X_test
        else:
            X = sm.add_constant(X_nuevo)
            
        prob = self.resultados.predict(X)
        return (prob > umbral).astype(int)
    
    def evaluar_modelo(self, umbral=0.5):
        y_pred = self.predecir(umbral)
        cm = confusion_matrix(self.y_test, y_pred)
        vp, fp, fn, vn = cm.ravel()
        
        return {
            'matriz_confusion': cm,
            'error_total': (fp + fn) / len(self.y_test),
            'sensibilidad': vp / (vp + fn),
            'especificidad': vn / (vn + fp)
        }
    
    def curva_roc(self):
        prob = self.resultados.predict(self.X_test)
        fpr, tpr, _ = roc_curve(self.y_test, prob)
        roc_auc = auc(fpr, tpr)
        
        plt.figure()
        plt.plot(fpr, tpr, label=f'Área bajo la curva = {roc_auc:.2f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('Tasa de Falsos Positivos')
        plt.ylabel('Tasa de Verdaderos Positivos')
        plt.title('Curva ROC')
        plt.legend()
        plt.show()
        
        return roc_auc