cd ~/590-PVB16/

DATE=$(date -Is)  
message="SYNC-"$DATE; #echo $message; exit
echo $message
git add ./; 
git commit -m $message; 
git push

