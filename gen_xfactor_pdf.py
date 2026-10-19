#!/usr/bin/env python3
"""Generates the X-Factor mathematics PDF (English, BSL 1.1).

Uses DejaVu Sans/Mono so the full math alphabet renders (ℝ, Σ, λ, ψ, subscripts).
"""
import os
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

DEJAVU_DIR = "/usr/share/fonts/truetype/dejavu"
pdfmetrics.registerFont(TTFont("DejaVu", os.path.join(DEJAVU_DIR, "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DejaVu-Bold", os.path.join(DEJAVU_DIR, "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("DejaVuMono", os.path.join(DEJAVU_DIR, "DejaVuSansMono.ttf")))
pdfmetrics.registerFont(TTFont("DejaVuMono-Bold", os.path.join(DEJAVU_DIR, "DejaVuSansMono-Bold.ttf")))


def build_pdf(path):
    doc = SimpleDocTemplate(path, pagesize=letter,
                            leftMargin=1 * inch, rightMargin=1 * inch,
                            topMargin=1 * inch, bottomMargin=1 * inch)
    styles = getSampleStyleSheet()
    title = ParagraphStyle("Title", parent=styles["Title"], alignment=TA_CENTER,
                           fontSize=16, fontName="DejaVu-Bold")
    sub = ParagraphStyle("Sub", parent=styles["Normal"], alignment=TA_CENTER, fontSize=10,
                         textColor=colors.grey, fontName="DejaVu")
    h1 = ParagraphStyle("H1", parent=styles["Heading1"], fontName="DejaVu-Bold", fontSize=13,
                        spaceBefore=12, spaceAfter=6, textColor=colors.black)
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontName="DejaVu-Bold", fontSize=11,
                        spaceBefore=8, spaceAfter=4, textColor=colors.black)
    body = ParagraphStyle("Body", parent=styles["BodyText"], fontSize=10, leading=14,
                          spaceAfter=6, fontName="DejaVu")
    mono = ParagraphStyle("Mono", parent=styles["Code"], fontSize=9.5, leading=14,
                          backColor=colors.whitesmoke, borderPadding=4,
                          fontName="DejaVuMono", spaceAfter=8, spaceBefore=4)

    story = []
    story.append(Paragraph("The X-Factor", title))
    story.append(Paragraph("Deterministic Manifold Embedding for Native LLM Inference", sub))
    story.append(Spacer(1, 6))
    story.append(Paragraph("Klenio Araujo Padilha &mdash; Winnex Brasil Soluções Empresariais LTDA-ME", sub))
    story.append(Paragraph("August 2026 &mdash; BSL 1.1 &mdash; Supplement to Zenodo 10.5281/zenodo.21861809", sub))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Abstract", h1))
    story.append(Paragraph(
        "The X-Factor is a <b>deterministic, training-free</b> operator that embeds arbitrary text into the "
        "latent space of a pretrained language model, enabling native inference without fine-tuning. It is "
        "computed <b>structurally</b> from a model's embedding matrix &mdash; never by running the model. "
        "This document gives the complete mathematics.", body))

    story.append(Paragraph("1. Problem Statement", h1))
    story.append(Paragraph(
        "A pretrained language model has an embedding matrix <b>E &isin; &#8477;<sup>V&times;D</sup></b> "
        "mapping V tokens to D-dimensional hidden vectors. To perform inference, text must be mapped to a "
        "sequence of hidden states in &#8477;<sup>D</sup>. The X-Factor constructs a deterministic projection "
        "from an arbitrary text representation into the model's latent space, using only the geometry of E.", body))

    story.append(Paragraph("2. The Manifold Hypothesis", h1))
    story.append(Paragraph(
        "The V rows of E do not fill &#8477;<sup>D</sup>; they concentrate near a lower-dimensional structure. "
        "Assume the token embeddings lie on a Riemannian manifold M embedded in &#8477;<sup>D</sup>. The X-Factor "
        "captures the linearized tangent structure of M &mdash; the principal subspace where embeddings concentrate.", body))
    story.append(Paragraph("2.1 Centering", h2))
    story.append(Paragraph(
        "The mean embedding is &#956; = (1/V) &Sigma;<sub>i=1..V</sub> E<sub>i</sub> &isin; &#8477;<sup>D</sup>.", body))
    story.append(Paragraph(
        "The centered matrix is <b>E&#772; = E &minus; &#956; &middot; 1<sup>T</sup> &isin; &#8477;<sup>V&times;D</sup></b>.", body))
    story.append(Paragraph("2.2 Covariance", h2))
    story.append(Paragraph(
        "The empirical covariance: <b>C = (1/V) E&#772;<sup>T</sup>E&#772; &isin; &#8477;<sup>D&times;D</sup></b>, "
        "symmetric positive semi-definite.", body))

    story.append(Paragraph("3. Eigendecomposition", h1))
    story.append(Paragraph("3.1 Spectral decomposition", h2))
    story.append(Paragraph(
        "C admits an orthonormal eigendecomposition <b>C = U &#923; U<sup>T</sup></b> where "
        "U = [u<sub>1</sub>, u<sub>2</sub>, &hellip;, u<sub>D</sub>] &isin; &#8477;<sup>D&times;D</sup> with "
        "U<sup>T</sup>U = I<sub>D</sub>, and "
        "&#923; = diag(&#955;<sub>1</sub>, &#955;<sub>2</sub>, &hellip;, &#955;<sub>D</sub>) with "
        "&#955;<sub>1</sub> &ge; &#955;<sub>2</sub> &ge; &hellip; &ge; &#955;<sub>D</sub> &ge; 0.", body))
    story.append(Paragraph("3.2 Computational method (power iteration with deflation)", h2))
    story.append(Paragraph(
        "The full eigendecomposition is O(D&sup3;). Following the Madhava principle &mdash; bound-guided pruning "
        "without exhaustive computation &mdash; the top-r eigenpairs are computed by <b>power iteration with "
        "deflation</b>:", body))
    story.append(Paragraph(
        "initialize v<sub>j</sub> (deterministic, unit norm)<br/>"
        "repeat:&nbsp; w = C&middot;v<sub>j</sub>;&nbsp; &#955;<sub>j</sub> = v<sub>j</sub><sup>T</sup>&middot;C&middot;v<sub>j</sub>"
        "&nbsp; (Rayleigh quotient);&nbsp; v<sub>j</sub> = w / &#8214;w&#8214;<br/>"
        "until |&#916;&#955;<sub>j</sub>| &lt; &#949;<br/>"
        "deflate:&nbsp; C &#8592; C &minus; &#955;<sub>j</sub>&middot;v<sub>j</sub>v<sub>j</sub><sup>T</sup>",
        mono))
    story.append(Paragraph(
        "Cost O(D&sup2;&middot;r&middot;iters) versus O(D&sup3;) &mdash; O(D/r) faster. The Rayleigh quotient plays the "
        "same role as the Cauchy-Schwarz bound in the Madhava engine: a provable selection criterion.", body))

    story.append(Paragraph("4. The Effective Rank", h1))
    story.append(Paragraph(
        "The effective rank at variance fraction &#964; is the smallest r such that "
        "&Sigma;<sub>k=1..r</sub> &#955;<sub>k</sub> &ge; &#964; &middot; &Sigma;<sub>k=1..D</sub> &#955;<sub>k</sub>, "
        "with &Sigma; &#955;<sub>k</sub> = tr(C). The X-Factor uses &#964; = 0.95.", body))

    story.append(Paragraph("5. The X-Factor Operator", h1))
    story.append(Paragraph("5.1 The basis", h2))
    story.append(Paragraph(
        "The X-Factor is <b>X = [u<sub>1</sub>, u<sub>2</sub>, &hellip;, u<sub>r</sub>] &isin; "
        "&#8477;<sup>D&times;r</sup></b>, the top-r principal directions.", body))
    story.append(Paragraph("5.2 The projector", h2))
    story.append(Paragraph(
        "The orthogonal projection onto the manifold subspace is <b>P = XX<sup>T</sup> &isin; "
        "&#8477;<sup>D&times;D</sup></b>.", body))
    story.append(Paragraph("<b>P&sup2; = P</b> (idempotent)&nbsp;&nbsp;and&nbsp;&nbsp;<b>P<sup>T</sup> = P</b> (symmetric).", mono))
    story.append(Paragraph(
        "For any h &isin; &#8477;<sup>D</sup>, &#8214;h &minus; Ph&#8214; is the distance to the manifold "
        "subspace, and Ph is the best r-rank approximation of h in the least-squares sense (Eckart&ndash;Young).", body))

    story.append(Paragraph("6. Text Embedding via the X-Factor", h1))
    story.append(Paragraph("6.1 Spectral tokenization (character histogram)", h2))
    story.append(Paragraph(
        "A text T = c<sub>1</sub>c<sub>2</sub>&hellip;c<sub>n</sub> maps to a deterministic feature vector, "
        "the character histogram:", body))
    story.append(Paragraph(
        "h<sub>j</sub>(T) = (1/&#8214;&middot;&#8214;) &middot; &Sigma;<sub>i=1..n</sub> "
        "1[ c<sub>i</sub> &#8801; j (mod B) ],&nbsp;&nbsp; j = 0..B&minus;1",
        mono))
    story.append(Paragraph(
        "normalized to unit L&sup2; norm. The histogram is chosen over a raw spectral average because the latter "
        "collapses (pairwise cosine between texts &#8776; 0.99); the histogram separates texts (cosine spread "
        "&#8776; 0.13&ndash;0.60).", body))
    story.append(Paragraph("6.2 Dimensionality expansion", h2))
    story.append(Paragraph(
        "The B-dimensional histogram is embedded into &#8477;<sup>D</sup> by zero-padding and L&sup2; "
        "normalization:", body))
    story.append(Paragraph(
        "&#968; = [h<sub>0</sub>, &hellip;, h<sub>B&minus;1</sub>, 0, &hellip;, 0] / &#8214;&middot;&#8214;", mono))
    story.append(Paragraph("6.3 Projection", h2))
    story.append(Paragraph(
        "The hidden state is <b>h = P&#968; = XX<sup>T</sup>&#968; &isin; &#8477;<sup>D</sup></b>, "
        "lying exactly on the model's embedding manifold.", body))

    story.append(Paragraph("7. The Complete Operator", h1))
    story.append(Paragraph(
        "<b>&#934;<sub>X</sub>(T) = X&middot;X<sup>T</sup> &middot; pad_L2( histogram(T) )</b>", mono))
    story.append(Paragraph(
        "<b>Training-free:</b> X depends only on E (a structural read). "
        "<b>Deterministic:</b> same text &rarr; same h. "
        "<b>Manifold-respecting:</b> &#934;<sub>X</sub>(T) &isin; span(X).", body))

    story.append(Paragraph("8. Complexity", h1))
    story.append(Paragraph(
        "Covariance C: O(VD&sup2;) or O(SD&sup2;) sampled. Power iteration: O(D&sup2;r&middot;iters). "
        "Projector P: O(D&sup2;r). Embedding a text: O(n + Dr). For D=2048, r&#8776;1700, S=10&#8308;: "
        "computed in ~73 s on one CPU.", body))

    story.append(Paragraph("9. Relation to the Madhava Engine", h1))
    story.append(Paragraph(
        "The X-Factor follows the Madhava design philosophy: <b>bound-guided selection</b> (the Rayleigh quotient as a "
        "Cauchy-Schwarz-style bound) instead of exhaustive computation; <b>deterministic, provable</b> operators; and "
        "<b>structural reads</b> of the model &mdash; no forward pass, no training.", body))

    story.append(Spacer(1, 16))
    story.append(Paragraph("BSL 1.1 &mdash; Business Source License. Contact: pay@winnex.ai", sub))

    doc.build(story)
    print(f"PDF written: {path}")


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    build_pdf(os.path.join(here, "XFACTOR_MATHEMATICS.pdf"))
