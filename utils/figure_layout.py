"""Compact headings for image comparison panels."""

from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea, VPacker


# Keep method names bold and metrics regular in either stacked or inline headings.
def panel_heading(axis, name: str, metric: str = "", fontsize: float = 8, multiline: bool = False):
    parts = [TextArea(name, textprops={"fontsize": fontsize, "fontweight": "bold"})]
    if metric or multiline:
        parts.append(TextArea(metric, textprops={"fontsize": fontsize}))
    content = (VPacker(children=parts, align="center", pad=0, sep=1) if multiline
               else HPacker(children=parts, align="baseline", pad=0, sep=3))
    heading = AnchoredOffsetbox(
        loc="lower center", child=content,
        frameon=False, pad=0, borderpad=0, bbox_to_anchor=(0.5, 1.025),
        bbox_transform=axis.transAxes,
    )
    axis.add_artist(heading)
    return heading
