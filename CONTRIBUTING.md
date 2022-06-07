# How to contribute

We welcome contributions from external contributors, and this document
describes how to merge code changes into this repository. 

## Getting Started

0. Make sure you have a [GitHub account](https://github.com/signup/free).
1. [Fork](https://help.github.com/articles/fork-a-repo/) this repository on GitHub.
2. On your local machine, [clone](https://help.github.com/articles/cloning-a-repository/) your fork of the repository.
3. [checkout](https://help.github.com/articles/creating-and-deleting-branches-within-your-repository/) a new branch of your local fork and give it a descriptive name: the feature it implements, the issue it addresses, etc.

## Making Changes

* Make some changes to the appropriate branch. Note that you should be working on a branch with a descriptive name, [*not the main branch*](http://blog.jasonmeridth.com/posts/do-not-issue-pull-requests-from-your-master-branch/)
* When you are ready for others to examine and comment on your new feature, navigate to your fork of `MolPAL` on GitHub and open a [pull request](https://help.github.com/articles/using-pull-requests/) (PR). Note that after you launch a PR from one of your fork's branches, all subsequent commits to that branch will be added to the open pull request automatically.  Each commit added to the PR will be validated for mergability, compilation and test suite compliance; the results of these tests will be visible on the PR page.
* When you're ready to be considered for merging, check the "Ready to go" box on the PR page to let the MolPAL devs know that the changes are complete. The code will not be merged until this box is checked, the continuous integration returns checkmarks, and multiple core developers give "Approved" reviews.

## Additional Resources

* [General GitHub documentation](https://help.github.com/)
* [PR best practices](http://codeinthehole.com/writing/pull-requests-and-other-good-practices-for-teams-using-github/)
* [A guide to contributing to software packages](http://www.contribution-guide.org)
* [Thinkful PR example](http://www.thinkful.com/learn/github-pull-request-tutorial/#Time-to-Submit-Your-First-PR)
