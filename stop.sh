echo 'call stop.sh'

# stop for one houses (main.py)
get_pids() {
 ps -f -U `whoami` | grep main.py | grep -v 'grep main.py' | grep -v 'stop.sh' | while read _USER_ _PID_ _OTHERS_ ; do
  echo $_PID_
 done
}

while true; do
 _PIDS_=`get_pids`
 if [ -z "$_PIDS_" ] ; then
  break
 fi
 echo kill $_PIDS_
 kill $_PIDS_
 sleep 1
done

echo '... done'