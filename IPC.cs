using GTA;
using IPC;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.IO;
using System.IO.MemoryMappedFiles;
using System.Linq;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace IPC
{
    enum FLAGS
    {
        REQUEST_GAME_STATE = 0,
        REQUEST_INPUT = 1,
        REQUEST_ACTION = 2,
        GAME_STATE_WRITTEN = 3,
        INPUT_WRITTEN = 4,
        ACTION_WRITTEN = 5,
        RESET = 6,
        IS_TRAINING = 7
    }

    public class Flags
    {
        public const int NUMBER_OF_FLAGS = 8;
        public const string FLAGS_TAG = "flags.ipc";

        private readonly MemoryMappedFile ipc_flags_f;
        public readonly MemoryMappedViewStream flags;
        public Flags(int n = NUMBER_OF_FLAGS, string tag = FLAGS_TAG)
        {
            int size = -(n / -8);
            ipc_flags_f = MemoryMappedFile.OpenExisting(tag);
            flags = ipc_flags_f.CreateViewStream(0, size);
        }

        public void SetFlag(int idx, bool value)
        {
            if (value) SetFlag(idx, 1);
            else SetFlag(idx, 0);
        }
        private void SetFlag(int idx, uint value)
        {
            int pos = idx / 8;
            int offset = idx % 8;
            uint mask = ~((uint)1 << offset);
            flags.Seek(pos, 0);
            uint state = (uint)flags.ReadByte();
            flags.Seek(pos, 0);
            byte updated_state = (byte)((state & mask) | (value << offset));
            flags.WriteByte(updated_state);
            flags.Flush();
        }
        public bool GetFlag(int idx)
        {
            int pos = idx / 8;
            int offset = idx % 8;
            int mask = 1 << offset;
            flags.Seek(pos, 0);
            int state = flags.ReadByte();
            return (state & mask) != 0;
        }

        public void WaitUntil(int idx, bool value, Action fn)
        {
            while (GetFlag(idx) != value) fn();
        }

        public void WaitUntil(int idx, bool value)
        {
            WaitUntil(idx, value, () => { });
        }

        public void WaitUntil(IEnumerable<int> idxs, IEnumerable<bool> values, Action fn)
        {
            while (idxs.Zip(values, (idx, value) => GetFlag(idx) == value).All(x => x))
                fn();
        }

        public void WaitUntil(IEnumerable<int> idxs, IEnumerable<bool> values)
        {
            WaitUntil(idxs, values, () => { });
        }
    }

    public class Channel
    {

        private readonly MemoryMappedFile ipc_f;
        private readonly MemoryMappedViewStream ipc;
        private readonly int size;

        public Channel(int size, string tag)
        {
            this.size = size;
            ipc_f = MemoryMappedFile.CreateOrOpen(tag, size);
            ipc = ipc_f.CreateViewStream(0, size);
        }

        public void Close()
        {
            ipc.Flush();
            ipc.Dispose();
            ipc_f.Dispose();
        }
        public void Put(byte[] payload)
        {
            ipc.Seek(0, 0);
            ipc.Write(payload, 0, payload.Length);
            ipc.Flush();
        }
        public byte[] Take()
        {
            byte[] payload = new byte[size];
            ipc.Seek(0, 0);
            ipc.Read(payload, 0, size);
            return payload;
        }
    }

    [AttributeUsage(AttributeTargets.Field, Inherited = true, AllowMultiple = false)]
    public class Mappable : Attribute
    {
        public int position;

        public Mappable(int position, int count = -1)
        {
            this.position = position;
        }
    }

    public class Map<T> where T : class, new()
    {
        public T map;
        private byte[] buffer = new byte[Length];

        private static readonly T defaultMap = new T();
        private static readonly FieldInfo[] memberFields = typeof(T).GetFields().Where(IsArrayField).OrderBy(field => GetAttribute(field).position).ToArray();

        private static readonly int[] lengths = memberFields.Select(field => Buffer.ByteLength((Array)field.GetValue(defaultMap))).ToArray();
        private static readonly int[] offsets = lengths.Select((_, index) => new ArraySegment<int>(lengths, 0, index).Sum()).ToArray();
        public static readonly int Length = lengths.Sum();

        public Map(T map)
        {
            this.map = map;
        }

        public Map()
        {
            this.map = new T();
        }

        private static bool IsArrayField(FieldInfo field)
        {
            return Attribute.GetCustomAttribute(field, typeof(Mappable)) != null && typeof(Array).IsAssignableFrom(field.FieldType);
        }
        private static Mappable GetAttribute(FieldInfo field)
        {
            return (Mappable)Attribute.GetCustomAttribute(field, typeof(Mappable));
        }

        public byte[] ToBytes(ref T map)
        {
            this.map = map;
            return bytes;
        }

        public T FromBytes(byte[] buffer)
        {
            bytes = buffer;
            return map;
        }
        public byte[] bytes
        {
            get
            {
                for (int i = 0; i < memberFields.Length; i++)
                    Buffer.BlockCopy((Array)memberFields[i].GetValue(map), 0, buffer, offsets[i], lengths[i]);
                return buffer;
            }

            set
            {
                value.CopyTo(buffer, 0);
                for (int i = 0; i < memberFields.Length; i++)
                    Buffer.BlockCopy(buffer, offsets[i], (byte[])memberFields[i].GetValue(map), 0, lengths[i]);
            }
        }
    }

    public class MappedChannel<T> : Channel where T : class, new()
    {

        public Map<T> map = new Map<T>();

        public void Put(ref T map)
        {
            this.map.map = map;
            Put(this.map.bytes);
        }
        public void Take(ref T map)
        {
            this.map.map = map;
            this.map.bytes = base.Take();
        }
        public new T Take()
        {
            T map = new T();
            this.map.map = map;
            this.map.bytes = base.Take();
            return map;
        }
        public MappedChannel(string tag) : base(Map<T>.Length, tag)
        {
        }
    }
}


public class GameState
{
    [Mappable(position: 0)]
    public float[] CAM = new float[3];

    [Mappable(position: 1)]
    public float[] VEL = new float[3];

    [Mappable(position: 2)]
    public int[] DMG = new int[1];
}

class GameIPC: Flags
{

    public MappedChannel<GameState> GameStateChannel = new MappedChannel<GameState>("game_state.ipc");
    
    public void Debug()
    {
        using (FileStream fsWrite = new FileStream("debug.txt", FileMode.OpenOrCreate, FileAccess.Write))
        {
            string[] debugstrs = new[] {
                $"REQUEST_GAME_STATE: {GetFlag((int) FLAGS.REQUEST_GAME_STATE)}",
                $"REQUEST_INPUT: {GetFlag((int) FLAGS.REQUEST_INPUT)}",
                $"REQUEST_ACTION: {GetFlag((int) FLAGS.REQUEST_ACTION)}",
                $"GAME_STATE_WRITTEN: {GetFlag((int) FLAGS.GAME_STATE_WRITTEN)}",
                $"INPUT_WRITTEN: {GetFlag((int) FLAGS.INPUT_WRITTEN)}",
                $"ACTION_WRITTEN: {GetFlag((int) FLAGS.ACTION_WRITTEN)}",
                $"RESET: {GetFlag((int) FLAGS.RESET)}",
                $"IS_TRAINING: {GetFlag((int) FLAGS.IS_TRAINING)}"
            };
            flags.Seek(0, 0);
            debugstrs = debugstrs.Append($"{flags.ReadByte()}").ToArray();
            string debugstr = string.Join("\n", debugstrs);
            byte[] bytestr = System.Text.Encoding.UTF8.GetBytes(debugstr);
            fsWrite.Write(bytestr, 0, bytestr.Length);
        }
    }
    public void WriteState(GameState state)
    {

        WaitUntil((int)FLAGS.REQUEST_GAME_STATE, true);
        GameStateChannel.Put(ref state);
        SetFlag((int)FLAGS.GAME_STATE_WRITTEN, true);
    }

    public void WriteStateImmediate(GameState state)
    {
        GameStateChannel.Put(ref state);
        SetFlag((int)FLAGS.GAME_STATE_WRITTEN, true);
    }

}