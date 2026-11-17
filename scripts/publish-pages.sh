#!/bin/sh
# Publish the built film to the gh-pages branch of this repository's `origin`.
# GitHub Pages then serves it at https://<user>.github.io/<repository>/ (Settings → Pages →
# "Deploy from a branch", branch gh-pages, folder / (root)).
#
# Nothing is published until someone runs this script and answers "yes".
#   npm run publish:pages
set -e

cd "$(dirname "$0")/.."

remote=$(git remote get-url origin 2>/dev/null || true)
if [ -z "$remote" ]; then
  echo "No remote named origin. Create the GitHub repository first, then:"
  echo "  git remote add origin git@github.com:<user>/piramid-observatory.git"
  exit 1
fi
if [ -n "$(git status --porcelain)" ]; then
  echo "The working tree has uncommitted changes. Commit them first, so the published film matches a commit."
  exit 1
fi

sha=$(git rev-parse --short HEAD)
npm run typecheck
npm run build
touch dist/.nojekyll

echo
echo "About to replace the gh-pages branch of"
echo "  $remote"
echo "with the build of commit $sha. This makes the film public."
printf "Type yes to publish: "
read answer
[ "$answer" = "yes" ] || { echo "Not published."; exit 1; }

# A throw-away repository that holds only the build, pushed as the single commit of gh-pages.
tmp=$(mktemp -d)
cp -R dist/. "$tmp"
cd "$tmp"
git init -q -b gh-pages
git add -A
git -c user.name="$(git -C "$OLDPWD" config user.name)" -c user.email="$(git -C "$OLDPWD" config user.email)" \
  commit -q -m "Publish $sha"
git push -f "$remote" gh-pages
echo "Published $sha. The build is in $tmp."
