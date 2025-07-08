
# github
ssh -T git@github.com 
 
git init

git add .

git commit -m "250616"

git remote add origin "git@github.com:LiebknechtKarl/Generate-RL.git"


git branch 250616
git checkout 250616

git push -u origin 250616
git pull --rebase origin 250616

git status

git add README.md


git commit -m "250616"
git rebase --continue

git pull --rebase origin 250616

git push origin 250616



