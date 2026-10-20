#!/bin/sh
# Fetch the robot models used by the simulations and add the palm-site copies.
# Run from the repo root.
set -e
MEN="${MENAGERIE_DIR:-models/mujoco_menagerie}"
if [ ! -d "$MEN/.git" ]; then
  mkdir -p "$(dirname "$MEN")"
  git clone https://github.com/google-deepmind/mujoco_menagerie.git "$MEN"
fi
git -C "$MEN" checkout 8161bba264d7fa7c99ca301e91e7fb44737676ad
cp spec/g1_hands_palm/g1_hands_palm.xml spec/g1_hands_palm/scene_hands_palm.xml "$MEN/unitree_g1/"
cp spec/h1_palm/h1_palm.xml spec/h1_palm/scene_palm.xml "$MEN/unitree_h1/"
echo "models ready in $MEN"
