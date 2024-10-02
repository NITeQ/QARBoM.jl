@echo off
REM Define the URL of the new repository (replace with your new repository URL)
set NEW_REPO_URL=https://github.com/username/new-repository.git

REM Define the path to the current (old) repository
set OLD_REPO=%cd%

REM Create a temporary directory to hold the copied files
set TEMP_DIR=%~dp0temp_copy

REM Clean up the temporary directory if it exists already
if exist "%TEMP_DIR%" rmdir /s /q "%TEMP_DIR%"

REM Create the temporary directory
mkdir "%TEMP_DIR%"

REM Use robocopy to copy all files and folders, excluding the "script" folder
robocopy "%OLD_REPO%" "%TEMP_DIR%" /e /xd "%OLD_REPO%\script"

REM Go to the temporary directory
cd /d %TEMP_DIR%

REM Initialize a new git repository in the temporary directory
git init

REM Add the remote URL for the new repository
git remote add origin %NEW_REPO_URL%

REM Add all files and commit them
git add .
git commit -m "Initial commit from old repository, excluding script folder"

REM Push to the new repository (you may need to provide credentials)
git branch -M main
git push -u origin main

REM Clean up the temporary directory
cd /d %~dp0
rmdir /s /q "%TEMP_DIR%"

REM Done
echo All done!
pause
