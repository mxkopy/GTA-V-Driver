using System;
using System.Collections.Generic;
using System.IO.MemoryMappedFiles;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Reflection;

internal class IPCChannel
{

    static MemoryMappedFile ipc_f;
    static MemoryMappedViewStream ipc;

    public IPCChannel(int length, string tag)
    {
        ipc_f = MemoryMappedFile.CreateOrOpen(tag, length);
        ipc = ipc_f.CreateViewStream(0, length);
    }

    public void Close()
    {
        ipc.Flush();
        ipc.Dispose();
        ipc_f.Dispose();
    }

    public bool IsPutLocked()
    {
        ipc.Seek(0, 0);
        return (0x1 & ipc.ReadByte()) == 1;
    }

    public bool IsConsumeLocked()
    {
        return !IsPutLocked();
    }

    public void SetLock(bool value)
    {
        // TODO: Just write bool?
        byte v = value ?  (byte) 0x1 : (byte) 0x0;
        ipc.Seek(0, 0);
        ipc.WriteByte(v);
        ipc.Flush();
    }

    public void UnlockConsume()
    {
        SetLock(true);
    }

    public void UnlockPut()
    {
        SetLock(false);
    }

    public void LockConsume()
    {
        UnlockPut();
    }

    public void LockPut()
    {
        UnlockConsume();
    }

    public void Put(byte[] payload)
    {
        ipc.Seek(1, 0);
        ipc.Write(payload, 0, payload.Length);
        ipc.Flush();
    }
    public byte[] Consume(long n=-1)
    {
        n = n == -1 ? ipc.Length : n;
        byte[] payload = new byte[n];
        ipc.Read(payload, 0, payload.Length);
        return payload;
    }

    public void PutBlocking(byte[] payload)
    {
        while (IsPutLocked());
        Put(payload);
    }

    public byte[] ConsumeBlocking(int n=-1)
    {
        while (IsConsumeLocked());
        return Consume(n);
    }
}

[AttributeUsage(AttributeTargets.Field, Inherited =true, AllowMultiple =false)]
public class IPCMappable: Attribute
{
    public int position;
    public int count;
    
    public IPCMappable(int position, int count)
    {
        this.position = position;
        this.count = count;
    }
}

class IPCMap<T>
{

    [IPCMappable(position: 0, count: 1)]
    int[] test;
    public IPCMappable[] MemberAttributes()
    {
        return MemberFields().Select(
            field => (IPCMappable) Attribute.GetCustomAttribute(field, typeof(T))
        ).ToArray();
    }

    public FieldInfo[] MemberFields()
    {
        return typeof(T).GetFields().Where(
            field => Attribute.GetCustomAttribute(field, typeof(T)) != null && typeof(Array).IsAssignableFrom(field.FieldType)
        ).ToArray();
    }

    public Array[] MemberArrays()
    {
        return MemberFields().Select(
            field =>
            {
                object fieldValue = field.GetValue(null);
                Array fieldArray = (Array)fieldValue;
                if (fieldArray.Length == 0)
                {
                    Type f = field.FieldType;
                    IPCMappable attribute = (IPCMappable)Attribute.GetCustomAttribute(field, typeof(IPCMappable));
                    fieldValue = Activator.CreateInstance(f, attribute.count);
                    field.SetValue(null, fieldValue);
                    fieldArray = (Array)fieldValue;
                }
                return fieldArray;
            }
        ).ToArray();
    }

    byte[] ToBytes()
    {
        int nb = MemberArrays().Sum(array => Buffer.ByteLength(array));
        byte[] bytes = new byte[nb];
        int[] positions = MemberAttributes().Select(attribute => attribute.position).ToArray();
        Array[] arrays = MemberArrays();
        Array.Sort(positions, arrays);
        int[] lengths = arrays.Select(array => Buffer.ByteLength(array)).ToArray();
        int[] starts = lengths.Select((_, index) => new ArraySegment<int>(lengths, 0, index).Sum()).ToArray();
        for (int i = 0; i < arrays.Length; i++)
        {
            int offset = new ArraySegment<int>(lengths, 0, i).Sum();
            Buffer.BlockCopy(arrays[i], 0, bytes, offset, lengths[i]);
        };
        return bytes;
    }
}

class GameIPCMap: IPCMap<GameIPCMap>
{
    [IPCMappable(position: 1, count: 3)]
    public static float[] CAM;

    [IPCMappable(position: 2, count: 3)]
    public static float[] VEL;

    [IPCMappable(position: 3, count: 1)]
    public static int DMG;

}

