import seaborn as sns

def build_colorpalette(palette, n_colors):
    palette = sns.color_palette(palette, n_colors)
    palette_plotly = []

    for r, g, b in palette:
        r, g, b = r * 256, g * 256, b * 256
        rgb = "rgb({},{},{})".format(r, g, b)
        palette_plotly.append(rgb)

    return palette_plotly
