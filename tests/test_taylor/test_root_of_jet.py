from probdiffeq import taylor
from probdiffeq.backend import np, tree


def test_root_of_jet(num=3):

    def vf(y):
        k1, k2, k3 = 0.04, 3e7, 1e4
        f0 = -k1 * y[0] + k3 * y[1] * y[2]
        f1 = k1 * y[0] - k2 * y[1] ** 2 - k3 * y[1] * y[2]
        f2 = k2 * y[1] ** 2
        return np.stack([f0, f1, f2])

    y0 = [np.asarray([1.0, 0.0, 0.0])]
    expected = taylor.odejet_unroll(vf, y0, num=num)

    def root(u, du, /):
        k1, k2, k3 = 0.04, 3e7, 1e4
        f0 = -k1 * u[0] + k3 * u[1] * u[2]
        f1 = k1 * u[0] - k2 * u[1] ** 2 - k3 * u[1] * u[2]
        return [du[0] - f0, du[1] - f1, u[0] + u[1] + u[2] - 1]

    is_differential = [np.asarray([True, True, False])]
    received = taylor.root_of_jet_of_root(
        root, y0, num=num, is_differential=is_differential
    )

    for a, b in zip(tree.tree_leaves(expected), tree.tree_leaves(received)):
        print("Expected", a)
        print("Received", b)

        print()
    assert False
