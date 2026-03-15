def discovered_law(inputs):
    m1 = inputs['m1']
    m2 = inputs['m2']
    r = inputs['r']
    G = 6.674e-5
    return G * m1 * m2 / (r ** 2)
