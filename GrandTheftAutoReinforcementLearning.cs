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
using IPC;

public class GrandTheftAutoReinforcementLearning : Script
{
    static readonly Vector3 AIRPORT = new Vector3(-1161.462f, -2584.786f, 13.505f);

    static GameIPC IPC = new GameIPC();
    static GameState State = new GameState();

    public GrandTheftAutoReinforcementLearning()
    {
        Tick += OnTick;
        KeyDown += OnKeyDown;

        Game.Player.Wanted.SetEveryoneIgnorePlayer(true);
        Game.Player.Wanted.SetPoliceIgnorePlayer(true);
    }
    private void OnTick(object sender, EventArgs e)
    {
        Vehicle V = Game.Player.Character.CurrentVehicle;

        Game.Player.Wanted.SetWantedLevel(0, false);
        Game.Player.Wanted.ApplyWantedLevelChangeNow(false);

        if (V != null)
        {
            if (IPC.GetFlag((int)FLAGS.IS_TRAINING))
            {
                State.CAM = Vector3.Project(GameplayCamera.Direction, V.ForwardVector).ToArray();
                State.VEL = Vector3.Project(V.Velocity, Vector3.Project(GameplayCamera.Direction, V.ForwardVector)).ToArray();
                State.DMG[0] = V.MaxHealth - V.Health;
                IPC.WriteState(State);
                while (IPC.GetFlag((int)FLAGS.GAME_STATE_WRITTEN) && IPC.GetFlag((int)FLAGS.IS_TRAINING));
            }
            else
            {
                Reset();
                //Yield();
                Wait(10);
            }
        }
    }

    private void Reset()
    {
        Game.Player.Wanted.SetEveryoneIgnorePlayer(true);
        Game.Player.Wanted.SetPoliceIgnorePlayer(true);
        if (Game.Player.Character.CurrentVehicle != null)
        {
            Game.Player.Character.CurrentVehicle.Delete();
        }
        Game.Player.Character.Position = AIRPORT;
        Vehicle vehicle = World.CreateVehicle(VehicleHash.EntityXF, Game.Player.Character.Position);
        Game.Player.Character.SetIntoVehicle(vehicle, VehicleSeat.Driver);
    }

    private void OnKeyDown(object sender, KeyEventArgs e)
    {
        if(e.KeyCode == Keys.Back)
        {
            Reset();
        }
    }
}