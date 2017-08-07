# lp - A Linear Programming Solver

Currently this is just scratching a personal itch of wanting a linear programming
solver but not wanting to figure out which of the common ones to interact with and
writing a translation layer for them. Furthermore, I've been sad that most (all?)
of them are either bad, GPL, or proprietary. Right now this one fits in the "bad"
category, but perhaps it will change in the future :).

It shouldn't be too hard to get oriented in the code if you're curious.
`builder.rs` is meant to be most program's interface with the solver, and
the structures there allow you to relatively easily define a problem and then
put it in standard form, as well as understand how to get the solution out
in your original terms. The types for standard form linear programs are located
in `problem.rs`. The only solver currently implemented is a very basic
simplex algorithm located in `simplex.rs`. I have done a very naive pivot selection
method, so I do not provide a termination guarantee. Again, this is something
that may change in the future :).

### Current Limitations

* Only simplex algorithm
* Pivot selection is naive, leading to possible loops or otherwise bad performance
* Matrix operations are done with dense matrices
* No information is preserved between LU factorizations in pivot steps
