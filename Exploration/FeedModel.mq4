#property copyright "Copyright 2017, MetaQuotes Software Corp."
#property link      "https://www.mql5.com"
#property version   "1.00"
#property strict
#include "../Include/NamedPipes.mqh"
#include "../Include/ShellCommands.mqh"

input int timeframe = 30;
input int bars_amount = 129;
input string suffix = "forex";
input int last_training_month = 1;
input int last_training_year = 2017;

int a = 0;
bool checkpoints[1];
string list_of_symbols[21];


int OnInit(){
   string argument = "C:/Users/louis/Documents/GitHub/ForexML/Exploration/NamedPipe.py " + suffix + " " + timeframe + " " + ArraySize(list_of_symbols);
   executeShellCommand("python", argument);
   
   // Initialisation
   list_of_symbols[0] = "EURUSD";
   list_of_symbols[1] = "GBPUSD";
   list_of_symbols[2] = "USDJPY";   
   list_of_symbols[3] = "USDCHF";
   list_of_symbols[4] = "AUDUSD";
   list_of_symbols[5] = "EURGBP";   
   list_of_symbols[6] = "USDCAD";
   list_of_symbols[7] = "EURJPY";
   list_of_symbols[8] = "GBPJPY";   
   list_of_symbols[9] = "AUDJPY";
   list_of_symbols[10] = "GBPAUD";
   list_of_symbols[11] = "GBPCAD";   
   list_of_symbols[12] = "EURAUD";
   list_of_symbols[13] = "CHFJPY";
   list_of_symbols[14] = "GBPCHF";   
   list_of_symbols[15] = "AUDCAD";
   list_of_symbols[16] = "CADJPY";
   list_of_symbols[17] = "CADCHF";   
   list_of_symbols[18] = "EURCHF";
   list_of_symbols[19] = "AUDCHF";
   list_of_symbols[20] = "EURCAD";
   checkpoints[0] = false;
/*
   
   string out;
   for (int j = 0; j < ArraySize(list_of_symbols); j++){
      out = out + list_of_symbols[j] + ",";
   }
   listenServer("MT4_Train", out);  
   Print(out); 
*/

   return(INIT_SUCCEEDED);
}

void OnDeinit(const int reason){
   listenServer("MT4_Train", "Done");
}
  
  
void OnTick(){
   datetime current_time = TimeLocal();
   if ((TimeMinute(current_time) >= 10) && (TimeDayOfWeek(current_time) > 1) && (checkpoints[0] == true))
      checkpoints[0] = false;
   // Checking that timestamp is correct
   else if ((TimeMinute(current_time) == 0) && (TimeDayOfWeek(current_time) > 1) && (TimeHour(current_time) <= 23) && (TimeHour(current_time) >= 0)&& (checkpoints[0] == false)){
         SendAllInfo(current_time);
   }
}

void SendAllInfo(datetime current_time){
   MqlRates rates[];
   ArraySetAsSeries(rates, true);
   for (int j = 0; j < ArraySize(list_of_symbols); j++){
      string out = j + ",";
      out = out + float(TimeHour(current_time)) + "|";
      int copied=CopyRates(list_of_symbols[j], timeframe, current_time, int(24*60/timeframe), rates);
      if (copied ==  int(60/timeframe*24)){ 
         out = out + rates[1].open + ",";
         out = out + rates[int(12*60/timeframe)].open + ",";
         out = out + rates[int(18*60/timeframe)].open + ",";
         out = out + rates[int(21*60/timeframe)].open + ",";
         out = out + rates[int(23*60/timeframe)].open + ",";
         out = out + FindHighestInRate(rates) + ",";
         out = out + FindLowestInRate(rates) + ",";
         out = out + FindTotalVolume(rates) + "|";
         copied=CopyRates(list_of_symbols[j], timeframe, current_time - (PERIOD_D1 * 60), bars_amount, rates);
         if(copied == bars_amount){
            for(int i=0;i<bars_amount;i++){
               out = out + rates[i].open + ",";
               out = out + (rates[i].high - rates[i].low) + ",";
               out = out + rates[i].tick_volume + "|";
            }
            listenServer("MT4_Train", out);
         }
      }
   }
   
   checkpoints[0] = true;
   a += 1;
   Print(a);
}

double FindHighestInRate(MqlRates& rates[]){
   double highest = 0;
   for(int i=0;i<ArraySize(rates);i++){
      if (rates[i].high > highest)
         highest = rates[i].high;
   }
   return highest;
}

double FindLowestInRate( MqlRates& rates[]){
   double lowest = 10000000;
   for(int i=0;i<ArraySize(rates);i++){
      if (rates[i].low < lowest)
         lowest = rates[i].low;
   }
   return lowest;
}


double FindTotalVolume(MqlRates& rates[]){
   double volumes = 0;
   for(int i=0;i<ArraySize(rates);i++)
      volumes += rates[i].tick_volume;
   return volumes;
}