# Content Credit: https://github.com/UtkarshPathrabe/Image-and-Video-Processing--From-Mars-to-Hollywood-with-a-stop-at-the-Hospital--Duke-University

# git clone https://github.com/UtkarshPathrabe/Image-and-Video-Processing--From-Mars-to-Hollywood-with-a-stop-at-the-Hospital--Duke-University.git
# mv "Image-and-Video-Processing--From-Mars-to-Hollywood-with-a-stop-at-the-Hospital--Duke-University/Weekly Quizzes" .
# rm -rf Image-and-Video-Processing--From-Mars-to-Hollywood-with-a-stop-at-the-Hospital--Duke-University

for i in $(seq 5); do
  cd "Quiz 0$i"
  grip Quiz0$i.md --export ../Quiz$i.html
  cd ..
done

for i in $(seq 6 8); do
  cd "Quiz 0$i"
  grip "Quiz 0$i.md" --export ../Quiz$i.html
  cd ..
done

for i in $(seq 8); do
  rm -r "Quiz 0$i"
done
