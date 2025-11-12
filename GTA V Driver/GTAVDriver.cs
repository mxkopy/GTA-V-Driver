using GTA;
using GTA.Math;
using GTA.NaturalMotion;
using System;
using System.Collections.Generic;
using System.IO;
using System.IO.MemoryMappedFiles;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;


// On start       <- Wait for NN ACK
//                -> NN sends random input to controller
//                -> Set NN ACK
//                -> Wait for GAME ACK
// Read NN ACK    <- Unset NN ACK
//                <- Get screenshot, game data, & user input; send NN input through. 
//                <- Set GAME ACK
//                <- Wait for NN ACK
// Read GAME ACK  -> Unset GAME ACK
//                -> Process screenshot, data. Predict user input. Send latest Ax <= b
//                -> Set NN ACK
//                -> Wait for GAME ACK

public enum GAME_PCKT
{
    ACK_END = 1,
    CAM_END = 13,
    VEL_END = 25,
    DMG_END = 29,
}





public class GTAVDriver : Script
{
    
    static MemoryMappedFile ipc;
    static MemoryMappedViewAccessor accessor;

    public GTAVDriver()
    {
        Tick += OnTick;
        ipc = MemoryMappedFile.CreateOrOpen("game.ipc", (long) GAME_PCKT.DMG_END);
        accessor = ipc.CreateViewAccessor(0, (long) GAME_PCKT.DMG_END);
    }

    private void OnTick(object sender, EventArgs e)
    {
        Vehicle V = Game.Player.Character.CurrentVehicle;
        Game.Player.WantedLevel = 0;

        if (V != null)
        {
            float[] CAM = Vector3.Project(GameplayCamera.Direction, V.ForwardVector).ToArray();
            float[] VEL = Vector3.Project(V.Velocity, Vector3.Project(GameplayCamera.Direction, V.ForwardVector)).ToArray();
            int DMG = V.MaxHealth - V.Health;
            V.Repair();
            accessor.WriteArray<float>((long) GAME_PCKT.ACK_END, CAM, 0, 3);
            accessor.WriteArray<float>((long) GAME_PCKT.CAM_END, VEL, 0, 3);
            accessor.Write<int>((long) GAME_PCKT.VEL_END, ref DMG);
            accessor.Write(0, 1);
            accessor.Flush();
            while (accessor.ReadByte(0) == 1)
            {
                Wait(1);
            };
        }
    }
}