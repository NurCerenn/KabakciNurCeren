# Feedback on the Git assignment

Any comments inserted directly into your files are marked `#CLAUDE>>` (written by
Claude, an AI) or `#DAN>>` (written by Dan). Find them all by searching for `>>`;
`grep -rn '>>' .` lists every one. They are ordinary code comments, so your code
runs exactly as it did before. Claude's comments carry no grade and Claude does
not grade; any grade for this assignment comes from Dan, at the end of his
section below.

## Claude Feedback

The hard part of this unit is done properly. Linh committed into your repo, you
have a real merge (`7ba92bc`, "solve conflict") where the resolved content differs
from both parents, and you did the rename chain `test.txt` → `notes.txt` →
`test1.txt` with git tracking every step as a rename. That's the whole
collaborate-conflict-resolve loop working end to end, which is what this unit is
actually for.

The thing to tighten is commit messages. Several of yours are just filenames —
"test2.txt", "notes.txt", "change". A filename tells a future reader *where* you
worked but not *what you did*, and git already knows where. Compare your own
"changes from linh's end", which is genuinely informative. The test to apply:
could you find this commit six months from now by searching the message? "renamed
notes.txt to test1.txt" passes; "notes.txt" doesn't. It costs nothing now and
saves real time once the scripts get long.

## Dan Feedback

I agree with Claude you could make better commit messages. Remember the git history will already say what files have been changed on a given commit, so the messages are meant to be more about what you did to them.

Good work.

Grade: S+