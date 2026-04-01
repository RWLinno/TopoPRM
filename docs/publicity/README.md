# 宣传与展示素材（终版）

## 页面入口

- `index.html`：统一入口（推荐）
- `demo_page.html`：动态总览页（支持 summary.json 自动读取/上传）
- `framework_visualization.html`：交互式框架图（点击高亮路径）
- `technical_roadmap.html`：技术路线图

## 宣讲稿

- `talk_track.md`：5-8 分钟可直接使用的讲稿

## 数据文件

- `data/summary.json`
- `data/summary.md`

## 刷新命令

```bash
cd /mnt/users/rwl/topoprm
python3 scripts/generate_publicity_report.py --eval_dir output/eval --output_dir docs/publicity/data
```
