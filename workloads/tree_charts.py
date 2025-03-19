import argparse
import json

import onnx
from pyecharts import options as opts
from pyecharts.charts import Tree
import tree

parser = argparse.ArgumentParser()
parser.add_argument("--model", "-m", type=str)
args = parser.parse_args()

model_name = args.model
model = onnx.load(model_name + ".onnx")

trees = tree.model2trees(model, None)

data = trees[0].toEchartsJSON()

c = (
    Tree(
        init_opts=opts.InitOpts(  # 图表画布大小，css长度单位
            width="100%",  # 宽度
            height="2000px",  # 高度
            # page_title="网页标题",
            # theme=ThemeType.LIGHT,  # 主题
        ),
    )
    .add(
        f"{model_name}",
        [data],
        pos_top="10%",
        pos_left="10%",
        pos_bottom="10%",
        pos_right="10%",
        is_roam=True,
        # width="100%",
        # height="100%",
        collapse_interval=0,
        orient="LR",
        label_opts=opts.LabelOpts(
            position="top",
            horizontal_align="right",
            vertical_align="middle",
            rotate=0,
        ),
    )
    .set_global_opts(
        title_opts=opts.TitleOpts(title="Tree"),
        datazoom_opts=opts.DataZoomOpts(is_zoom_on_mouse_wheel="alt"),
    )
    .render(f"{model_name}.html")
)
