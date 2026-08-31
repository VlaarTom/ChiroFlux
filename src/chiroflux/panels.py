"""Section titles used to group CLI options in ``--help``.

Typer renders each ``rich_help_panel`` as its own bordered panel, in the order
the panels first appear in a command's signature. Keeping the titles here
means the commands present their options in the same order with the same
names, and that a reader comparing two commands can find the same flag in the
same place.

Declare a command's options grouped in this order, skipping any panel that
does not apply (a single-simulation command has no symmetry corrections):

    INPUT -> DATASET -> SELECT -> REPR -> SYMMETRY -> MODEL -> OUTPUT

This module imports nothing, so any analysis module can use it.
"""

#: Where the simulation lives and which column is the order parameter.
INPUT = "Input data"

#: Which paths become rows, and how the two simulations are reconciled.
DATASET = "Dataset construction"

#: Which CV columns survive, and what they are called.
SELECT = "CV selection"

#: Changes to how a CV is represented. Applied to every simulation alike.
REPR = "CV corrections: representation"

#: Symmetry corrections that undo a convention difference between two
#: simulations. The title carries the warning because applying one of these
#: to both simulations cancels it out.
SYMMETRY = "CV corrections: symmetry (apply to ONE simulation only)"

#: Estimator choice, cross-validation, hyperparameters, parallelism.
MODEL = "Model and training"

#: Where results are written.
OUTPUT = "Output"
