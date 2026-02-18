patientFolder = '/home/subhajitroy005/Documents/Projects/ECG/PTB_DB/patient001';
recordName    = 's0010_re';

[signal, Fs, tm] = ptbdb_get_signal(patientFolder, recordName);

plot(tm, signal(:,1))


