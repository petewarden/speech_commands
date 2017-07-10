package org.tensorflow.speechcommands;

import android.content.Intent;
import android.content.pm.PackageManager;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.net.Uri;
import android.os.Environment;
import android.support.annotation.NonNull;
import android.support.design.widget.FloatingActionButton;
import android.support.v4.app.ActivityCompat;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.support.design.widget.Snackbar;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

import static java.lang.Math.min;

public class MainActivity extends AppCompatActivity {

    private static final int SAMPLE_RATE = 16000;
    private static final int SAMPLE_DURATION_MS = 2000;
    private static final int RECORDING_LENGTH = (int) (SAMPLE_RATE * SAMPLE_DURATION_MS / 1000);
    private static final long AVERAGE_WINDOW_DURATION_MS = 500;
    private static float DETECTION_THRESHOLD=0.70f;
    private static int SUPPRESSION_MS = 1500;
    private static int MINIMUM_COUNT = 3;
    private static long MINIMUM_TIME_BETWEEN_SAMPLES_MS = 30;
    private static String LABEL_FILENAME = "file:///android_asset/low_latency_conv_labels.txt";
    private static String MODEL_FILENAME = "file:///android_asset/conv_v1_frozen.pb";
    private static String INPUT_DATA_NAME = "decoded_sample_data:0";
    private static String SAMPLE_RATE_NAME = "decoded_sample_data:1";
    private static String OUTPUT_SCORES_NAME = "labels_softmax";

    private static final int REQUEST_RECORD_AUDIO = 13;
    private TextView mInfo;
    private TextView mCurrentCommand;
    private TextView mPreviousCommand;
    private Button mQuitButton;
    private static final String LOG_TAG = MainActivity.class.getSimpleName();

    short[] mRecordingBuffer = new short[RECORDING_LENGTH];
    short[] mSpeechInput = new short[RECORDING_LENGTH];
    int mRecordingOffset = 0;
    boolean mShouldContinue = true;
    private Thread mRecordingThread;
    boolean mShouldContinueRecognition = true;
    private Thread mRecognitionThread;
    private final ReentrantLock mRecordingBufferLock = new ReentrantLock();
    private TensorFlowInferenceInterface mInferenceInterface;
    private List<String> mLabels = new ArrayList<String>();
    private class RecognitionResult {
        public ArrayList<Float> mResults = new ArrayList<>();
        public long mCaptureTime;
    }
    private ArrayDeque<RecognitionResult> mRecognitionResults = new ArrayDeque<>();
    private RecognizeCommands mRecognizeCommands = null;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        mInfo = (TextView) findViewById(R.id.textView);
        mCurrentCommand = (TextView) findViewById(R.id.currentCommand);
        mPreviousCommand = (TextView) findViewById(R.id.previousCommand);
        mQuitButton = (Button) findViewById(R.id.quit);
        mQuitButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                moveTaskToBack(true);
                android.os.Process.killProcess(android.os.Process.myPid());
                System.exit(1);
            }
        });

        String actualFilename = LABEL_FILENAME.split("file:///android_asset/")[1];
        Log.i(LOG_TAG, "Reading labels from: " + actualFilename);
        BufferedReader br = null;
        try {
            br = new BufferedReader(new InputStreamReader(getAssets().open(actualFilename)));
            String line;
            while ((line = br.readLine()) != null) {
                mLabels.add(line);
            }
            br.close();
        } catch (IOException e) {
            throw new RuntimeException("Problem reading label file!" , e);
        }

        mRecognizeCommands = new RecognizeCommands(mLabels, AVERAGE_WINDOW_DURATION_MS,
                DETECTION_THRESHOLD, SUPPRESSION_MS, MINIMUM_COUNT, MINIMUM_TIME_BETWEEN_SAMPLES_MS);

        mInferenceInterface = new TensorFlowInferenceInterface(getAssets(), MODEL_FILENAME);

        requestMicrophonePermission();
        startRecognition();
    }

    private void requestMicrophonePermission() {
        if (ActivityCompat.shouldShowRequestPermissionRationale(this, android.Manifest.permission.RECORD_AUDIO)) {
            // Show dialog explaining why we need record audio
            Snackbar.make(mInfo, "Microphone access is required in order to record audio",
                    Snackbar.LENGTH_INDEFINITE).setAction("OK", new View.OnClickListener() {
                @Override
                public void onClick(View v) {
                    ActivityCompat.requestPermissions(MainActivity.this, new String[]{
                            android.Manifest.permission.RECORD_AUDIO}, REQUEST_RECORD_AUDIO);
                }
            }).show();
        } else {
            ActivityCompat.requestPermissions(MainActivity.this, new String[]{
                    android.Manifest.permission.RECORD_AUDIO}, REQUEST_RECORD_AUDIO);
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions,
                                           @NonNull int[] grantResults) {

        if (requestCode == REQUEST_RECORD_AUDIO && grantResults.length > 0 &&
                grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            startRecording();
            startRecognition();
        }
    }

    public void startRecording() {
        if (mRecordingThread != null)
            return;

        mShouldContinue = true;
        mRecordingThread = new Thread(new Runnable() {
            @Override
            public void run() {
                record();
            }
        });
        mRecordingThread.start();
    }

    public void stopRecording() {
        if (mRecordingThread == null)
            return;

        mShouldContinue = false;
        mRecordingThread = null;
    }

    private void record() {
        android.os.Process.setThreadPriority(android.os.Process.THREAD_PRIORITY_AUDIO);

        // buffer size in bytes
        int bufferSize = AudioRecord.getMinBufferSize(SAMPLE_RATE,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT);

        if (bufferSize == AudioRecord.ERROR || bufferSize == AudioRecord.ERROR_BAD_VALUE) {
            bufferSize = SAMPLE_RATE * 2;
        }

        short[] audioBuffer = new short[bufferSize / 2];

        AudioRecord record = new AudioRecord(MediaRecorder.AudioSource.DEFAULT,
                SAMPLE_RATE,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT,
                bufferSize);

        if (record.getState() != AudioRecord.STATE_INITIALIZED) {
            Log.e(LOG_TAG, "Audio Record can't initialize!");
            return;
        }

        record.startRecording();

        Log.v(LOG_TAG, "Start recording");

        while (mShouldContinue) {
            int numberRead = record.read(audioBuffer, 0, audioBuffer.length);
            int maxLength = mRecordingBuffer.length;
            int newRecordingOffset = mRecordingOffset + numberRead;
            int secondCopyLength = Math.max(0, newRecordingOffset - maxLength);
            int firstCopyLength = numberRead - secondCopyLength;
            mRecordingBufferLock.lock();
            try {
                System.arraycopy(audioBuffer, 0, mRecordingBuffer, mRecordingOffset, firstCopyLength);
                System.arraycopy(audioBuffer, firstCopyLength, mRecordingBuffer, 0, secondCopyLength);
                mRecordingOffset = newRecordingOffset % maxLength;
            } finally {
                mRecordingBufferLock.unlock();
            }
        }

        record.stop();
        record.release();
    }

    public void startRecognition() {
        if (mRecognitionThread != null)
            return;

        mShouldContinueRecognition = true;
        mRecognitionThread = new Thread(new Runnable() {
            @Override
            public void run() {
                recognize();
            }
        });
        mRecognitionThread.start();
    }

    public void stopRecognition() {
        if (mRecognitionThread == null)
            return;

        mShouldContinueRecognition = false;
        mRecognitionThread = null;
    }

    private void recognize() {
        Log.v(LOG_TAG, "Start recognition");

        short[] inputBuffer = new short[RECORDING_LENGTH];
        float[] floatInputBuffer = new float[RECORDING_LENGTH];
        float[] outputScores = new float[mLabels.size()];
        String[] outputScoresNames = new String[] {OUTPUT_SCORES_NAME};
        int[] sampleRateList = new int[] {SAMPLE_RATE};

        while (mShouldContinueRecognition) {
            mRecordingBufferLock.lock();
            try {
                int maxLength = mRecordingBuffer.length;
                int firstCopyLength = maxLength - mRecordingOffset;
                int secondCopyLength = mRecordingOffset;
                System.arraycopy(mRecordingBuffer, mRecordingOffset, inputBuffer, 0, firstCopyLength);
                System.arraycopy(mRecordingBuffer, 0, inputBuffer, firstCopyLength, secondCopyLength);
            } finally {
                mRecordingBufferLock.unlock();
            }

            for (int i = 0; i < RECORDING_LENGTH; ++i) {
                floatInputBuffer[i] = inputBuffer[i] / 32767.0f;
            }
            mInferenceInterface.feed(SAMPLE_RATE_NAME, sampleRateList);
            mInferenceInterface.feed(INPUT_DATA_NAME, floatInputBuffer, RECORDING_LENGTH, 1);
            mInferenceInterface.run(outputScoresNames);
            mInferenceInterface.fetch(OUTPUT_SCORES_NAME, outputScores);

            long currentTime = System.currentTimeMillis();
            String localLabelOutput;
            String localLabelOnly;
            if (false) {
                RecognitionResult recognitionResult = new RecognitionResult();
                for (int i = 0; i < mLabels.size(); ++i) {
                    recognitionResult.mResults.add(outputScores[i]);
                }
                recognitionResult.mCaptureTime = currentTime;
                mRecognitionResults.addFirst(recognitionResult);

                while (true) {
                    RecognitionResult oldResult = mRecognitionResults.peekLast();
                    long age = currentTime - oldResult.mCaptureTime;
                    if (age > AVERAGE_WINDOW_DURATION_MS) {
                        mRecognitionResults.removeLast();
                    } else {
                        break;
                    }
                }

                ArrayList<Float> averageScores = new ArrayList<Float>();
                for (int i = 0; i < mLabels.size(); ++i) {
                    averageScores.add(0.0f);
                }
                for (Iterator<RecognitionResult> itr = mRecognitionResults.iterator(); itr.hasNext(); ) {
                    RecognitionResult result = itr.next();
                    for (int i = 0; i < result.mResults.size(); ++i) {
                        float oldAverage = averageScores.get(i);
                        averageScores.set(i, oldAverage + result.mResults.get(i) / mRecognitionResults.size());
                    }
                }

                float topScore = 0.0f;
                String topLabel = "";
                for (int i = 0; i < mLabels.size(); ++i) {
                    if (averageScores.get(i) > topScore) {
                        topScore = averageScores.get(i);
                        topLabel = mLabels.get(i);
                    }
                }
                localLabelOutput = topLabel + ":" + topScore;
                localLabelOnly = topLabel;
            } else {
                RecognizeCommands.RecognitionResult result = mRecognizeCommands.processLatestResults(outputScores, currentTime);
                if (result.mIsNewCommand) {
                    localLabelOutput = result.mFoundCommand + ":" + result.mScore;
                    localLabelOnly = result.mFoundCommand;
                } else {
                    localLabelOutput = result.mFoundCommand + ":" + result.mScore;
                    localLabelOnly = "_silence_";
                }
            }
            final String labelOutput = localLabelOutput;
            final String labelOnly = localLabelOnly;
            //Log.v(LOG_TAG, labelOutput);
            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    mInfo.setText(labelOutput);
                    String oldCurrent = mCurrentCommand.getText().toString();
                    if ((labelOnly.charAt(0) != '_') && (!oldCurrent.equals(labelOnly))) {
                        mCurrentCommand.setText(labelOnly);
                        mPreviousCommand.setText(oldCurrent);
                    }
                }
            });
        }

        Log.v(LOG_TAG, "End recognition");
    }

}
