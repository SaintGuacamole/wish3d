using System.Collections.Generic;
using System.Text;
using UnityEngine;
using UnityEngine.XR.MagicLeap;
using UnityEngine.UI;
using System.Collections;
using System.IO;
using UnityEngine.Networking;
using Dummiesman;

public class VoiceIntentsExample : MonoBehaviour
{
    [SerializeField, Tooltip("Configuration file that holds list of voice commands.")] 
    private MLVoiceIntentsConfiguration _voiceConfiguration;
    private AudioClip clip;
    private byte[] bytes;
    private string fastApiUrl = "https://curvy-meals-yawn.loca.lt/generate-mesh";
    // private AudioSource audioSource;

    private readonly MLPermissions.Callbacks permissionCallbacks = new MLPermissions.Callbacks();

    private void Start()
    {
         //Permission Callbacks
        permissionCallbacks.OnPermissionGranted += OnPermissionGranted;
        permissionCallbacks.OnPermissionDenied += OnPermissionDenied;
        permissionCallbacks.OnPermissionDeniedAndDontAskAgain += OnPermissionDenied;
        
        // Requests permissions from the user. 
        MLPermissions.RequestPermission(MLPermission.VoiceInput, permissionCallbacks);
    }

    private void OnPermissionDenied(string permission)
    {
        Debug.Log("Permission Denied!");
    }

    private void OnPermissionGranted(string permission)
    {
        Initialize();
    }

    // Start the voice intent service with the configured voice commands.
    private void Initialize()
    {
        // audioSource = gameObject.AddComponent<AudioSource>();
        MLVoice.OnVoiceEvent += VoiceEvent;

        if (MLVoice.VoiceEnabled)
        {
            MLResult result = MLVoice.SetupVoiceIntents(_voiceConfiguration);
            if (result.IsOk)
            {
                    // Subscribe to the voice command event
                    MLVoice.OnVoiceEvent += VoiceEvent;
            }
        }
    }

    // Called when a voice command is detected.
    void VoiceEvent(in bool wasSuccessful, in MLVoice.IntentEvent voiceEvent)
    {
        StringBuilder strBuilder = new StringBuilder();
        strBuilder.Append($"<b>Last Voice Event:</b>\n");
        strBuilder.Append($"Was Successful: <i>{wasSuccessful}</i>\n");
        strBuilder.Append($"State: <i>{voiceEvent.State}</i>\n");
        // strBuilder.Append($"No Intent Reason\n(Expected NoReason): \n<i>{voiceEvent.NoIntentReason}</i>\n");
        // strBuilder.Append($"Event Unique Name:\n<i>{voiceEvent.EventName}</i>\n");
        // strBuilder.Append($"Event Unique Id: <i>{voiceEvent.EventID}</i>\n");
        
        Debug.Log(strBuilder.ToString());
        Debug.Log("start recording");
        clip = Microphone.Start(null, false, 10, 16000);
        StartCoroutine(StopRecording());
    }

    // Stop the service and disable the event when the script is destroyed.
    private void OnDestroy()
    {
        MLVoice.Stop();
        MLVoice.OnVoiceEvent -= VoiceEvent;

        permissionCallbacks.OnPermissionGranted -= OnPermissionGranted;
        permissionCallbacks.OnPermissionDenied -= OnPermissionDenied;
        permissionCallbacks.OnPermissionDeniedAndDontAskAgain -= OnPermissionDenied;
    }

    private IEnumerator StopRecording()
    {
        yield return new WaitForSeconds(10);
        var position = Microphone.GetPosition(null);
        Microphone.End(null);
        Debug.Log("Stop recording");
        var samples = new float[position * clip.channels];
        clip.GetData(samples, 0);
        bytes = EncodeAsWAV(samples, clip.frequency, clip.channels);
        Debug.Log("sending request to server! Hold on!");
        StartCoroutine(SendAudioAndGetOBJ(bytes));
        // audioSource.clip = clip;
        // audioSource.Play();

    }

    private byte[] EncodeAsWAV(float[] samples, int frequency, int channels)
    {
        using (var memoryStream = new MemoryStream(44 + samples.Length * 2))
        {
            using (var writer = new BinaryWriter(memoryStream))
            {
                writer.Write("RIFF".ToCharArray());
                writer.Write(36 + samples.Length * 2);
                writer.Write("WAVE".ToCharArray());
                writer.Write("fmt ".ToCharArray());
                writer.Write(16);
                writer.Write((ushort)1);
                writer.Write((ushort)channels);
                writer.Write(frequency);
                writer.Write(frequency * channels * 2);
                writer.Write((ushort)(channels * 2));
                writer.Write((ushort)16);
                writer.Write("data".ToCharArray());
                writer.Write(samples.Length * 2);

                foreach (var sample in samples)
                {
                    writer.Write((short)(sample * short.MaxValue));
                }
            }
            return memoryStream.ToArray();
        }
    }

    private IEnumerator SendAudioAndGetOBJ(byte[] audioData) {
        Debug.Log("preparing to send line 134");
        UnityWebRequest www = UnityWebRequest.PostWwwForm(fastApiUrl, "POST");
        
        WWWForm form = new WWWForm();
        form.AddBinaryData("audio", audioData, "recording.wav", "audio/wav");
        www.uploadHandler = new UploadHandlerRaw(form.data);
        www.downloadHandler = new DownloadHandlerBuffer();
        
        Debug.Log("preparing to send line 142");
        foreach (var header in form.headers)
        {
            www.SetRequestHeader(header.Key, header.Value);
        }

        Debug.Log("sending web request");
        yield return www.SendWebRequest();

        if (www.result != UnityWebRequest.Result.Success)
        {
            Debug.LogError("API Error: " + www.error);
        }
        else
        {
            // Handle the received OBJ data
            byte[] receivedObjBytes = www.downloadHandler.data;
            HandleOBJ(receivedObjBytes);
        }
    }

     void HandleOBJ(byte[] objData) {
        Debug.Log("Received OBJ file of size: " + objData.Length + " bytes.");

        // Save the byte array to a temporary file
        string tempFilePath = Path.Combine(Application.temporaryCachePath, "tempObj.obj");
        File.WriteAllBytes(tempFilePath, objData);

        // Load the OBJ from the temporary file path using Dummiesman OBJLoader
        GameObject loadedObj = new OBJLoader().Load(tempFilePath);
        loadedObj.transform.SetParent(this.transform); // Set as a child of the current game object

        // Change the scale of the loaded object
        float scaleFactor = 0.1f;  // Change this value to suit your needs
        loadedObj.transform.localScale = new Vector3(scaleFactor, scaleFactor, scaleFactor);

        // Optionally delete the temp file if no longer needed
        File.Delete(tempFilePath);
    }
}
