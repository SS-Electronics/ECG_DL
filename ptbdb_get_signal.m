% Get the DB File information patient wise


function [signal, Fs, tm] = ptbdb_get_signal(patientFolder, recordName)

    % Save current directory
    oldFolder = pwd;

    % Go to data folder
    cd(patientFolder);

    % Read ECG record (no extension!)
    [signal, Fs, tm] = rdsamp(recordName);

    % Return to original directory
    cd(oldFolder);

end

