import numpy as np
import colour
import matplotlib.pyplot as plt
RGB = np.random.random((32, 32, 3))
print(RGB)
colour.plotting.plot_RGB_chromaticities_in_chromaticity_diagram_CIE1931(
    [[[]]], colourspaces=['sRGB'])
plt.show()
