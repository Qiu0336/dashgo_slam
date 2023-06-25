#include "keyborad.h"

int KeyboardInput::scanKeyboard()
{
  int in;
  tcgetattr(STDIN_FILENO,&stored_settings);
  new_settings = stored_settings;
  new_settings.c_lflag &= (~ICANON);
  new_settings.c_cc[VTIME] = 0;
  tcgetattr(STDIN_FILENO,&stored_settings);
  new_settings.c_cc[VMIN] = 1;
  tcsetattr(STDIN_FILENO,TCSANOW,&new_settings);
  in = getchar();
  tcsetattr(STDIN_FILENO,TCSANOW,&stored_settings);
  return in;
}

void KeyboardInput::Run()
{
  while(1)
  {
    int i = scanKeyboard();
    SetKey(i);
    if(i == 'q' || i == 'Q')
      break;
  }
}

int KeyboardInput::GetKey()
{
    unique_lock<mutex> lock(mutexkey);
    int _key = key;
    key = 0;
    return _key;
}
int KeyboardInput::SetKey(int _key)
{
    unique_lock<mutex> lock(mutexkey);
    key = _key;
}


