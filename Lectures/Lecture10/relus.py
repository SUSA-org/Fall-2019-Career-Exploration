import numpy as np

def train(x, y, num_relus=20, mode='nonlinear'):
    LEARNING_RATE = 0.01

    a = np.random.random(size=num_relus) - 0.5
    b = np.random.random(size=num_relus) - 0.5
    c = np.random.random(size=num_relus) - 0.5

    def L(y, y_hat):
        return (y - y_hat)**2

    def R(x):
        if mode == 'nonlinear':
            return np.maximum(0, b + c*x)
        else:
            return b + c*x

    def f(x):
        return np.dot(a, R(x))

    def backward(x, y):
        nonlocal a, b, c
        
        y_hat = f(x)
        dy_hat = -2*(y - y_hat)

        R_hat = R(x)

        dR = a

        da = R_hat
        if mode == 'nonlinear':
            db = np.zeros(num_relus)
            db[R_hat > 0] = 1
        else:
            db = np.ones(num_relus)

        dc = np.zeros(num_relus)
        if mode == 'nonlinear':
            dc[R_hat > 0] = x
        else:
            dc.fill(x)
            
        a -= LEARNING_RATE * dy_hat * R_hat
        b -= LEARNING_RATE * dy_hat * dR * db
        c -= LEARNING_RATE * dy_hat * dR * dc

    # SGD
    ITERATIONS = 1000

    for _ in range(ITERATIONS):
        i = np.random.choice(x.shape[0])
        x_i = x[i]
        y_i = y[i]
        y_hat = f(x_i)
        backward(x_i, y_i)
    
    return f