"""Self-contained, printable academic HTML report (open in a browser → Print → Save as PDF)."""
import html
import re


def _md(text: str) -> str:
    text = html.escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    return re.sub(r"`(.+?)`", r"<code>\1</code>", text)


def _table(df, caption, n, fmt="{:.4f}"):
    def cell(v):
        return fmt.format(v) if isinstance(v, float) else html.escape(str(v))
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in [df.index.name or ""] + list(df.columns))
    body = "".join("<tr><td>" + html.escape(str(i)) + "</td>" + "".join(f"<td>{cell(v)}</td>" for v in row)
                   + "</tr>" for i, row in zip(df.index, df.itertuples(index=False)))
    return (f"<figure class='tbl'><figcaption><b>Table {n}.</b> {_md(caption)}</figcaption>"
            f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></figure>")


def _div(fig):
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"displaylogo": False})


def F(fig, caption, n):
    return f"<figure class='fig'>{_div(fig)}<figcaption><b>Figure {n}.</b> {_md(caption)}</figcaption></figure>"


CSS = """
@page{size:A4;margin:22mm 18mm}
body{font-family:Georgia,'Times New Roman',serif;color:#1c2430;max-width:960px;margin:auto;padding:28px;line-height:1.55}
.title{text-align:center;border-top:4px solid #1F3A5F;border-bottom:1px solid #ccd;padding:26px 0 18px;margin-bottom:22px}
.title .inst{font-variant:small-caps;letter-spacing:.06em;color:#556}
.title h1{font-size:24px;color:#1F3A5F;margin:12px 40px}
.title .sub{font-style:italic;color:#334;margin:-4px 0 10px}
.title .meta{color:#334}
h2{color:#1F3A5F;border-bottom:1px solid #dde;padding-bottom:4px;margin-top:30px;font-size:20px}
h3{color:#1F3A5F;font-size:16px;margin-bottom:4px}
.abstract{background:#f6f8fb;border-left:4px solid #1F3A5F;padding:12px 18px;text-align:justify}
p,li{text-align:justify}
table{border-collapse:collapse;width:100%;font-size:12.5px;font-family:'Segoe UI',Arial,sans-serif}
thead th{border-top:2px solid #222;border-bottom:1px solid #222;padding:5px 8px;text-align:right}
tbody td{padding:4px 8px;text-align:right;border-bottom:1px solid #eee}
th:first-child,td:first-child{text-align:left}
tbody tr:last-child td{border-bottom:2px solid #222}
figure{margin:16px 0 22px}
figcaption{font-size:13px;color:#334;margin:6px 0}
.tbl figcaption{margin-bottom:6px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:4px 14px}
.flow{display:flex;flex-wrap:wrap;gap:6px;align-items:center;justify-content:center;font-size:12px}
.flow span{border:1px solid #1F3A5F;background:#f3f5f8;border-radius:6px;padding:6px 9px;text-align:center}
.flow i{color:#1F3A5F;font-style:normal}
code{font-size:12px;background:#f1f3f6;padding:1px 4px;border-radius:3px}
.note{font-size:12px;color:#556}
@media print{body{padding:0}h2{break-after:avoid}figure,table{break-inside:avoid}.pb{break-before:page}}
"""


def build_report(project, info, features, labels, best, abstract, data_text, class_def, desc, models, test_tbl,
                 cv_tbl, sig, robust, discussion, limitations, conclusion, future, figs) -> str:
    """robust = {"text": str, "table": DataFrame, "fig": Figure} for the spectrum-only analysis."""
    from plotly.offline import get_plotlyjs
    e = html.escape
    b = test_tbl.loc[best]
    flow = " <i>→</i> ".join(f"<span>{s}</span>" for s in [
        f"Sensor data<br>{info['n_rows']} records", "Sieve labelling<br>(oil temperature)", f"Stratified split<br>"
        f"{info['n_train']} / {info['n_test']}", "IQR clipping +<br>z-score", f"{info['cv_folds']}-fold<br>CV",
        "Train 8<br>models", "Hold-out<br>evaluation", "Repeat without<br>temperature"])
    cms = "".join(f"<div>{_div(f)}</div>" for f in figs["cms"])

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>{e(project.get('title', 'Results report'))}</title><style>{CSS}</style>
<script>{get_plotlyjs()}</script></head><body>

<div class="title"><div class="inst">{e(project.get('institution', ''))}</div>
<h1>{e(project.get('title', ''))}</h1>
<div class="sub">{e(project.get('subtitle', ''))}</div>
<div class="meta">{e(project.get('degree', ''))} · {e(project.get('year', ''))}<br>
<b>Candidate:</b> {e(project.get('author', ''))} &nbsp;|&nbsp; <b>Supervisors:</b> {e(project.get('supervisor', ''))}</div></div>

<h2>Abstract</h2><div class="abstract">{_md(abstract)}</div>

<h2>1. Problem definition and data</h2>
<p>{_md(data_text)}</p>
<div class="grid">{_table(class_def, "Oil-condition classes, labelling rule and sample distribution.", 1, "{}")}
{F(figs['classes'], f"Samples per class (N = {info['n_rows']}).", 1)}</div>
{_table(desc, "Descriptive statistics of the input attributes.", 2, "{:.3f}")}

<h2 class="pb">2. Methodology</h2>
<figure><div class="flow">{flow}</div><figcaption><b>Figure 2.</b> Experimental pipeline. All fitted
transformations are learned from training data only.</figcaption></figure>
<h3>2.1 Pre-processing</h3>
<p>Records with missing values are removed. Each attribute is clipped to [Q1 − 1.5·IQR, Q3 + 1.5·IQR] (EDA outlier
treatment) and standardised with the z-score. Quartiles, means and standard deviations are estimated on the training
set; within cross-validation they are re-estimated on every training fold to avoid information leakage.</p>
<h3>2.2 Models</h3>
{_table(models, "Classifiers evaluated and their configuration.", 3, "{}")}
<p class="note">Neural networks: Adam (learning rate 10<sup>−3</sup>), {info['epochs']} epochs, batch size
{info['batch_size']}, categorical cross-entropy, 20% of the training data held out for validation.</p>
<h3>2.3 Evaluation protocol</h3>
<p>Models are compared by {info['cv_folds']}-fold stratified cross-validation on the training set and on a stratified
held-out test set ({info['test_size']:.0%}, n = {info['n_test']}, seed {info['seed']}) used only for the final
evaluation. Reported metrics are accuracy, macro-averaged precision, recall and F1-score, Cohen's κ, the multi-class
Matthews correlation coefficient (MCC) and the one-vs-rest macro ROC-AUC. Pairwise differences are tested with
McNemar's exact test (α = 0.05). The whole procedure is run twice: with all 13 attributes (thesis protocol) and
without the temperature attribute (spectrum-only protocol).</p>

<h2 class="pb">3. Results — thesis protocol</h2>
<p>{e(best)} obtained the best test-set performance: accuracy {b['Accuracy']:.2%}, macro F1-score
{b['F1-score']:.2%} and ROC-AUC {b['ROC-AUC']:.3f} (Table 4).</p>
{_table(test_tbl, f"Performance on the held-out test set (n = {info['n_test']}), sorted by macro F1-score.", 4)}
{_table(cv_tbl, f"{info['cv_folds']}-fold stratified cross-validation on the training set (mean ± SD).", 5, "{}")}
{F(figs['comparison'], f"Macro F1-score in {info['cv_folds']}-fold cross-validation (bars, ±1 SD) and on the "
                       f"held-out test set (diamonds).", 3)}
{F(figs['roc'], f"One-vs-rest ROC curves of {best} on the test set.", 4)}
<figure><div class="grid">{cms}</div><figcaption><b>Figure 5.</b> Test-set confusion matrices (rows: true class,
columns: predicted class; counts and row percentages).</figcaption></figure>
{F(figs['per_class'], "F1-score per class and model on the test set.", 6)}
{_table(sig, f"McNemar's exact test: {best} compared with every other model on the test set.", 6)}
{F(figs['gap'], "Training vs test accuracy; a large gap indicates over-fitting.", 7)}

<h2 class="pb">4. Robustness analysis — spectrum only</h2>
<p>{_md(robust['text'])}</p>
{_table(robust['table'], "Test-set accuracy and macro F1-score with and without the temperature attribute.", 7)}
{F(robust['fig'], "Test-set macro F1-score of every model under both protocols.", 8)}

<h2 class="pb">5. Discussion</h2>
{''.join(f"<p><b>{e(h)}.</b> {_md(t)}</p>" for h, t in discussion)}
<h3>Limitations</h3><ul>{''.join(f"<li>{_md(x)}</li>" for x in limitations)}</ul>

<h2>6. Conclusion</h2><div class="abstract">{_md(conclusion)}</div>
<h3>Future work</h3><ul>{''.join(f"<li>{_md(x)}</li>" for x in future)}</ul>
<p class="note">Models trained {e(info['trained_at'])}. All numbers in this report are computed directly from the
saved models and data splits.</p>
</body></html>"""
