import time

from lectrace import Reference, link, note, table, text

from _helpers import dose_chart

BUDGET_MG = 4000


def main():
    text("# Writing one of these")
    text(
        "A lecture is a plain Python file. Define `main()` at the top, put the helper "
        "functions below it, call `main()` at the bottom. That is the only structural "
        "rule."
    )
    text(
        """
        from lectrace import text

        def main():
            text("# Paracetamol dosing")
            doses = [500, 1000, 500]      # @inspect doses
            total = sum(doses)            # @inspect total

        if __name__ == "__main__":
            main()
        """,
        verbatim=True,
    )
    text(
        "Tracing starts when `main()` is called, so imports and `def` statements "
        "never produce a step. A helper function shows up in the viewer only if "
        "`main()` actually calls it."
    )

    text("## The five directives")
    text(
        "Directives are ordinary comments. They are the whole configuration surface "
        "of the tool."
    )

    text("### @inspect puts a variable in the panel")
    doses = [500, 1000, 500]  # @inspect doses
    total = sum(doses)  # @inspect total
    text(
        "New variables arrive in green. Watch `total` turn amber on the next line, "
        "which is how a changed value announces itself."
    )
    total = total + 500  # @inspect total
    text(
        "Inside `main()` you only see what you asked for. That is deliberate: a "
        "lecture is an argument, and the panel should hold the four numbers the "
        "argument is about, not everything in scope."
    )

    text("### @clear takes one back out")
    headroom = BUDGET_MG - total  # @inspect headroom
    text("Once `doses` has done its job, drop it and keep the panel readable.")
    safe = headroom > 0  # @clear doses @inspect safe

    text("### @stepover runs a call without descending into it")
    text(
        "By default the tracer follows every call into its body. When the body is "
        "plumbing rather than the point, say so:"
    )
    checked = validate_dose(total)  # @stepover @inspect checked

    text("### @hide runs a line but keeps it off the screen")
    text(
        "Useful for seeding a random number generator, or for the two lines of setup "
        "that would break the flow. The line still executes. There is one on the next "
        "line of this file and you cannot see it."
    )
    started = time.time()  # @hide

    text("### @invisible does the same and produces no step at all")
    _ = started  # @invisible

    text("## Inside a helper, you get everything for free")
    text(
        "The rule flips once you leave `main()`. Every local is shown without any "
        "directive, and the call stack appears above the panel so you can see where "
        "you are. Step into this one instead of over it."
    )
    schedule = build_schedule(total)  # @inspect schedule
    text(
        "This is the part that replaces `print(x.shape)`. You write the function the "
        "way you would write it anyway, and you get to walk through it."
    )
    link(build_schedule)
    dose_chart(schedule)  # @stepover
    text(
        "That chart came from `_helpers.py`. A file whose name starts with an "
        "underscore is imported like any other module and never traced, which is "
        "where the plumbing goes so the step-through stays on the argument."
    )

    text("## Everything you can put on the page")
    table(
        [
            {"Call": "text(...)", "Renders": "markdown, with LaTeX between $ signs"},
            {"Call": "text(..., verbatim=True)", "Renders": "monospace, whitespace preserved"},
            {"Call": "note(...)", "Renders": "a speaker note, styled as a callout"},
            {"Call": "table(rows, headers, caption)", "Renders": "a table from dicts, lists, or a DataFrame"},
            {"Call": "plot(spec)", "Renders": "a matplotlib figure, or a Vega-Lite dict for an interactive chart"},
            {"Call": "image(path, width)", "Renders": "a local file or a remote URL, inlined"},
            {"Call": "video(path, width)", "Renders": "an embedded player"},
            {"Call": "link(some_function)", "Renders": "a jump to that function in the viewer"},
            {"Call": "link(url_reference(arxiv_url))", "Renders": "a citation card, fetched from arXiv"},
            {"Call": "link(Reference(...))", "Renders": "a citation card you fill in yourself"},
            {"Call": "system_text([...])", "Renders": "the output of a shell command"},
        ],
        caption="File 01 uses every one of these. Read its source for the examples.",
    )

    text("## Getting it in front of people")
    text(
        """
        uv add lectrace
        uv run lectrace serve            # localhost:7000, reloads on save
        uv run lectrace serve 01_attention.py
        uv run lectrace build            # static site in _site/
        uv run lectrace init             # GitHub Actions workflow
        git push                         # deploys to GitHub Pages
        """,
        verbatim=True,
    )
    text(
        "`lectrace init` writes a workflow file. After that, every push to `main` "
        "rebuilds the site and publishes it, and the link you hand somebody keeps "
        "working without you doing anything."
    )
    text(
        "Files are numbered to order the sidebar. Both files here share the same "
        "`_helpers.py`."
    )
    text(
        """
        01_attention.py
        02_write_your_own.py
        """,
        verbatim=True
    )

    text("## Using it on your own work")
    table(
        [
            {"Instead of": "a screenshot of a loss curve",
             "Do this": "step through the training loop that produced it, one iteration at a time"},
            {"Instead of": "adding print(x.shape) to the forward pass",
             "Do this": "mark the tensor with @inspect and watch its shape in the Variables panel"},
            {"Instead of": "a slide asserting that softmax normalises",
             "Do this": "show one row before and after, then step backwards so people can compare"},
            {"Instead of": "a README that describes the pipeline",
             "Do this": "a page where the reader watches the pipeline run, line by line"},
        ],
        caption="The tool is not the interesting part. The habit is.",
    )
    text(
        "Writing an explanation of a computation is a harder test of understanding "
        "than producing the computation. With a coding assistant sitting next to you, "
        "it is close to the only honest one left. That is the reason to reach for "
        "this rather than a screenshot."
    )
    note(
        "Close here. Offer to take any file they are already working on and put it "
        "through lectrace live, which is a better ending than a summary slide."
    )

    link(
        Reference(
            title="lectrace",
            authors=["Dayo PraiseGod"],
            url="https://praisegee.github.io/lectrace/",
            description="Documentation, the full rendering API, and the deployment guide.",
        )
    )
    link(title="edtrace", url="https://github.com/percyliang/edtrace", authors=["Percy Liang"])


def validate_dose(total):
    time.sleep(0.01)
    return total <= BUDGET_MG


def build_schedule(total):
    """Split a daily total into four doses, rounding to something a nurse can pour."""
    per_dose = total / 4
    rounded = round(per_dose / 250) * 250
    schedule = []
    for hour in [8, 14, 20, 2]:
        schedule.append({"hour": hour, "mg": rounded})
    return schedule


if __name__ == "__main__":
    main()
