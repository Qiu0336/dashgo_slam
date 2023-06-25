#pragma once

#include <termio.h>
#include <stdio.h>
#include <unistd.h>
#include <iostream>
#include <mutex>
using namespace std;

class KeyboardInput
{
public:
    KeyboardInput(){};

    int scanKeyboard();

    void Run();

    int GetKey();
    int SetKey(int _key);

    mutex mutexkey;
    int key;
    struct termios new_settings;
    struct termios stored_settings;
};



