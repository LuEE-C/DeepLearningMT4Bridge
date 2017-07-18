//+------------------------------------------------------------------+
//|                                                ShellCommands.mqh |
//|                        Copyright 2017, MetaQuotes Software Corp. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2017, MetaQuotes Software Corp."
#property link      "https://www.mql5.com"
#property strict
#import "shell32.dll"
   int ShellExecuteW(int hWnd, string Verb, string File, string Parameter, string Path, int ShowCommand);
#import

int executeShellCommand(string file, string parameters=""){
   int r = ShellExecuteW(NULL, NULL, file, parameters, NULL, 5);
   if (r <= 32){   Alert("Shell failed: ", r); return(false);  }
   return(true);  
}