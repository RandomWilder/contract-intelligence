[Setup]
AppName=Contract Intelligence Platform
AppVersion=1.0.0
AppPublisher=Your Company
AppPublisherURL=https://yourwebsite.com
DefaultDirName={autopf}\ContractIntelligence
DefaultGroupName=Contract Intelligence Platform
OutputDir=installers
OutputBaseFilename=ContractIntelligence-Setup-Windows
Compression=lzma
SolidCompression=yes
WizardStyle=modern
SetupIconFile=icon.ico
UninstallDisplayIcon={app}\ContractIntelligence.exe

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "dist\ContractIntelligence\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\Contract Intelligence Platform"; Filename: "{app}\ContractIntelligence.exe"
Name: "{autodesktop}\Contract Intelligence Platform"; Filename: "{app}\ContractIntelligence.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\ContractIntelligence.exe"; Description: "{cm:LaunchProgram,Contract Intelligence Platform}"; Flags: nowait postinstall skipifsilent
