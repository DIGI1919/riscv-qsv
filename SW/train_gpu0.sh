 
sourceDir="bufferdir1"

sourceDir="bufferdir1"
targetDir="xiao_both_res18_c10"
f="0" 
exp="res18_Image" 

scriptDir="$( cd "$( dirname "$0" )" && pwd )"
mainRoot="$(realpath "$scriptDir/..")"
saveRoot="$mainRoot/save"
finalTargetDir="$saveRoot/$targetDir"
fullSourceDir="$scriptDir/$sourceDir"

mkdir -p "$saveRoot"
mkdir -p "$finalTargetDir"
mkdir -p "$fullSourceDir"

python -u "$mainRoot/print.py" \
    -d="$fullSourceDir" \
    -f="$f" \
    --exp_config="$exp" \
    2>&1 | tee "$finalTargetDir/$targetDir.log"

if [ -d "$fullSourceDir" ]; then
    mv "$fullSourceDir"/* "$finalTargetDir"/ 2>/dev/null
else
    echo "Warning: sourceDir not found."
fi
