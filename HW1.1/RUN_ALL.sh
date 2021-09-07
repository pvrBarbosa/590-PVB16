ScriptLoc=${PWD}
cd LectureCodes
for i in *.py; do echo $i; python $i; done #run all python scripts in directory
grep "I HAVE WORKED" *
cd $ScriptLoc #return to script directory
for i in *.py; do echo $i; python $i; done #run all python scripts in directory

