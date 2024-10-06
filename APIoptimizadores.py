import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

# Nuevo título de la aplicación
st.title("Simulación de Algoritmos de Optimización")

# Inicializar variables en el estado de sesión
if 'sim_data' not in st.session_state:
    st.session_state.sim_data = None 

if 'graph_3d' not in st.session_state:
    st.session_state.graph_3d = None  

# Contenedor principal
with st.container():
    left_col, right_col = st.columns(2)

    # Columna izquierda para la gráfica 3D
    with left_col:
        st.header("Visualización 3D")
        lower_limit = st.number_input("Definir límite inferior", value=-6.5)
        upper_limit = st.number_input("Definir límite superior", min_value=(lower_limit + 0.1), value=6.5)
        st.session_state.bounds = (lower_limit, upper_limit)

        if st.button("Generar Gráfico"):
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(8, 8))
            X = np.arange(lower_limit, upper_limit, 0.25)
            Y = np.arange(lower_limit, upper_limit, 0.25)
            X, Y = np.meshgrid(X, Y)
            R = np.sqrt(X**2 + Y**2)
            Z = -np.sin(R)
            
            # Cambiar la paleta de colores
            surface = ax.plot_surface(X, Y, Z, cmap=cm.viridis, linewidth=0, antialiased=True)
            ax.set_zlim(-1.01, 1.01)
            ax.zaxis.set_major_locator(LinearLocator(10))
            ax.zaxis.set_major_formatter('{x:.02f}')
            fig.colorbar(surface, shrink=0.5, aspect=5)
            st.session_state.graph_3d = fig  

        # Mostrar la gráfica si está disponible
        if st.session_state.graph_3d is not None:
            st.pyplot(st.session_state.graph_3d)

    # Columna derecha para los algoritmos de optimización
    with right_col:
        def compute_gradient(x, y):
            R = np.sqrt(x**2 + y**2)
            grad_x = -np.cos(R) * (x / R)
            grad_y = -np.cos(R) * (y / R)
            return np.array([grad_x, grad_y])
        
        def gradient_descent(theta, iterations, learning_rate):
            for _ in range(iterations):
                x, y = theta
                grad = compute_gradient(x, y)
                theta -= learning_rate * grad
            return theta
        
        def stochastic_gd(theta, train_data, iterations, learning_rate):
            for _ in range(iterations):
                np.random.shuffle(train_data)
                for point in train_data:
                    x, y = point
                    grad = compute_gradient(x, y)
                    theta -= learning_rate * grad
            return theta
        
        def rmsprop_optimizer(theta, train_data, iterations, learning_rate, decay_rate, epsilon):
            E_g2 = np.zeros_like(theta)
            for _ in range(iterations):
                np.random.shuffle(train_data)
                for point in train_data:
                    x, y = point
                    grad = compute_gradient(x, y)
                    E_g2 = decay_rate * E_g2 + (1 - decay_rate) * grad**2
                    theta -= learning_rate / (np.sqrt(E_g2) + epsilon) * grad
            return theta
        
        def adam_optimizer(theta, train_data, iterations, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
            m = np.zeros_like(theta)
            v = np.zeros_like(theta)
            t = 0
            
            for _ in range(iterations):
                np.random.shuffle(train_data)
                for point in train_data:
                    x, y = point
                    t += 1
                    grad = compute_gradient(x, y)
                    
                    m = beta1 * m + (1 - beta1) * grad
                    v = beta2 * v + (1 - beta2) * (grad**2)
                    
                    m_hat = m / (1 - beta1**t)
                    v_hat = v / (1 - beta2**t)
                    theta -= alpha * m_hat / (np.sqrt(v_hat) + epsilon)
            return theta

        # Sección para los métodos de optimización
        st.header("Optimización")
        lower_limit, upper_limit = st.session_state.bounds
        optimizer = st.selectbox("Selecciona un algoritmo:", ["", "Descenso de Gradiente", "Descenso de Gradiente Estocástico", "RMSPROP", "Adam"])

        if optimizer == "Descenso de Gradiente":
            theta_init = st.number_input("Valor inicial de theta", value=2.0)
            learn_rate = st.number_input("Tasa de aprendizaje", min_value=0.001, value=0.1)
            num_iter = st.number_input("Número de iteraciones", min_value=1, value=1000)
            theta_vals = np.array([float(theta_init), float(theta_init)])

            if st.button("Ejecutar Optimización"):
                final_theta = gradient_descent(theta_vals, num_iter, learn_rate)
                st.write(f"Valor mínimo estimado: {final_theta}")

        elif optimizer == "Descenso de Gradiente Estocástico":
            theta_init = st.number_input("Valor inicial de theta", value=2.0)
            learn_rate = st.number_input("Tasa de aprendizaje", min_value=0.001, value=0.01)
            num_iter = st.number_input("Número de iteraciones", min_value=1, value=100)
            num_points = st.number_input("Número de puntos de datos", min_value=1, value=100)
            theta_vals = np.array([float(theta_init), float(theta_init)])

            np.random.seed(42)
            x_data = np.random.uniform(lower_limit, upper_limit, num_points)
            y_data = np.random.uniform(lower_limit, upper_limit, num_points)
            train_data = list(zip(x_data, y_data))
            
            if st.button("Ejecutar Optimización"):
                final_theta = stochastic_gd(theta_vals, train_data, num_iter, learn_rate)
                st.write(f"Valor mínimo estimado: {final_theta}")

        elif optimizer == "RMSPROP":
            theta_init = st.number_input("Valor inicial de theta", value=2.0)
            learn_rate = st.number_input("Tasa de aprendizaje", min_value=0.001, value=0.001)
            num_iter = st.number_input("Número de iteraciones", min_value=1, value=100)
            decay = st.number_input("Factor de decaimiento", min_value=0.0, value=0.9)
            num_points = st.number_input("Número de puntos de datos", min_value=1, value=100)
            epsilon = st.number_input("Valor de epsilon", min_value=0.0, value=1e-8)
            theta_vals = np.array([float(theta_init), float(theta_init)])

            np.random.seed(42)
            x_data = np.random.uniform(lower_limit, upper_limit, num_points)
            y_data = np.random.uniform(lower_limit, upper_limit, num_points)
            train_data = list(zip(x_data, y_data))
            
            if st.button("Ejecutar Optimización"):
                final_theta = rmsprop_optimizer(theta_vals, train_data, num_iter, learn_rate, decay, epsilon)
                st.write(f"Valor mínimo estimado: {final_theta}")

        elif optimizer == "Adam":
            theta_init = st.number_input("Valor inicial de theta", value=2.0)
            num_iter = st.number_input("Número de iteraciones", min_value=1, value=100)
            alpha = st.number_input("Valor de alpha", min_value=0.0, value=0.001)
            beta1 = st.number_input("Valor de beta1", min_value=0.0, value=0.9)
            beta2 = st.number_input("Valor de beta2", min_value=0.0, value=0.999)
            epsilon = st.number_input("Valor de epsilon", min_value=0.0, value=1e-8)
            num_points = st.number_input("Número de puntos de datos", min_value=1, value=100)
            theta_vals = np.array([float(theta_init), float(theta_init)])

            np.random.seed(42)
            x_data = np.random.uniform(lower_limit, upper_limit, num_points)
            y_data = np.random.uniform(lower_limit, upper_limit, num_points)
            train_data = list(zip(x_data, y_data))
            
            if st.button("Ejecutar Optimización"):
                final_theta = adam_optimizer(theta_vals, train_data, num_iter, alpha, beta1, beta2, epsilon)
                st.write(f"Valor mínimo estimado: {final_theta}")
