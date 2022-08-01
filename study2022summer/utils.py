import seaborn as sns
import plotly.graph_objects as go

symbols = {"IP1": "circle", "IP2": "square", "ISS1": "triangle-up", "ISS2": "x"}
dashes = {2: None, 3: "dot"}


def build_colorpalette(palette, n_colors):
    palette = sns.color_palette(palette, n_colors)
    palette_plotly = []

    for r, g, b in palette:
        r, g, b = r * 256, g * 256, b * 256
        rgb = "rgb({},{},{})".format(r, g, b)
        palette_plotly.append(rgb)

    return palette_plotly


def plot_sdri(
    fig,
    data,
    symbol="circle",
    dash=None,
    label="",
    color="black",
    width=3,
    marker_size=12,
):
    x, y = data["times"], data["sdri"]

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines+markers",
            name=label,
            marker={
                "size": marker_size,
                "symbol": symbol,
            },
            line={"color": color, "width": width, "dash": dash},
        )
    )


def box_plot_sdri(
    fig,
    data,
    label="",
):
    fig.add_trace(go.Box(y=data, name=label))
