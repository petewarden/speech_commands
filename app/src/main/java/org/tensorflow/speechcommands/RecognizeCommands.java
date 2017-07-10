package org.tensorflow.speechcommands;

import android.util.Log;
import android.util.Pair;
import java.nio.FloatBuffer;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Deque;
import java.util.List;

public class RecognizeCommands {
    // Configuration settings.
    private List<String> mLabels = new ArrayList<String>();
    private long mAverageWindowDurationMS;
    private float mDetectionThreshold;
    private int mSuppressionMS;
    private int mMinimumCount;
    private long mMinimumTimeBetweenSamplesMS;

    // Working variables.
    private Deque<Pair<Long, float[]>> mPreviousResults = new ArrayDeque<Pair<Long, float[]>>();
    private String mPreviousTopLabel;
    private int mLabelsCount;
    private long mPreviousTopLabelTime;
    private float mPreviousTopLabelScore;

    public RecognizeCommands(List<String> labels, long averageWindowDurationMS,
                             float detectionThreshold, int suppressionMS, int minimumCount,
                             long minimumTimeBetweenSamplesMS) {
        mLabels = labels;
        mAverageWindowDurationMS = averageWindowDurationMS;
        mDetectionThreshold = detectionThreshold;
        mSuppressionMS = suppressionMS;
        mMinimumCount = minimumCount;
        mLabelsCount = mLabels.size();
        mPreviousTopLabel = "_silence_";
        mPreviousTopLabelTime = Long.MIN_VALUE;
        mPreviousTopLabelScore = 0.0f;
        mMinimumTimeBetweenSamplesMS = minimumTimeBetweenSamplesMS;
    }

    public class RecognitionResult {
        public final String mFoundCommand;
        public final float mScore;
        public final boolean mIsNewCommand;

        public RecognitionResult(String foundCommand, float score, boolean isNewCommand) {
            mFoundCommand = foundCommand;
            mScore = score;
            mIsNewCommand = isNewCommand;
        }
    }

    private class ScoreForSorting implements Comparable<ScoreForSorting> {
        public final float mScore;
        public final int mIndex;

        public ScoreForSorting(float score, int index) {
            mScore = score;
            mIndex = index;
        }
        @Override
        public int compareTo(ScoreForSorting other) {
            if (this.mScore > other.mScore) {
                return -1;
            } else if (this.mScore < other.mScore) {
                return 1;
            } else {
                return 0;
            }
        }
    }

    public RecognitionResult processLatestResults(float[] currentResults, long currentTimeMS) {
        if (currentResults.length != mLabelsCount) {
            throw new RuntimeException("The results for recognition should contain " + mLabelsCount +
                    " elements, but there are " + currentResults.length);
        }

        if ((!mPreviousResults.isEmpty()) &&
                (currentTimeMS < mPreviousResults.getFirst().first)) {
            throw new RuntimeException(
                    "You must feed results in increasing time order, but received a timestamp of " +
                    currentTimeMS + " that was earlier than the previous one of " +
                    mPreviousResults.getFirst().first);
        }

        final int howManyResults = mPreviousResults.size();
        // Ignore any results that are coming in too frequently.
        if (howManyResults > 1) {
            final long timeSinceMostRecent = currentTimeMS - mPreviousResults.getLast().first;
            if (timeSinceMostRecent < mMinimumTimeBetweenSamplesMS) {
                return new RecognitionResult(mPreviousTopLabel, mPreviousTopLabelScore, false);
            }
        }

        // Add the latest results to the head of the queue.
        mPreviousResults.addLast(new Pair<Long, float[]>(currentTimeMS, currentResults));

        // Prune any earlier results that are too old for the averaging window.
        final long timeLimit = currentTimeMS - mAverageWindowDurationMS;
        while (mPreviousResults.getFirst().first < timeLimit) {
            mPreviousResults.removeFirst();
        }

        // If there are too few results, assume the result will be unreliable and
        // bail.
        final long earliestTime = mPreviousResults.getFirst().first;
        final long samplesDuration = currentTimeMS - earliestTime;
        if ((howManyResults < mMinimumCount) ||
                (samplesDuration < (mAverageWindowDurationMS / 4))) {
            Log.v("RecognizeResult", "Too few results");
            return new RecognitionResult(mPreviousTopLabel, 0.0f, false);
        }

        // Calculate the average score across all the results in the window.
        float[] averageScores = new float[mLabelsCount];
        for (Pair<Long, float[]> previousResult : mPreviousResults) {
            final float[] scoresTensor = previousResult.second;
            int i = 0;
            while(i < scoresTensor.length) {
                averageScores[i] += scoresTensor[i] / howManyResults;
                ++i;
            }
        }

        // Sort the averaged results in descending score order.
        ScoreForSorting[] sortedAverageScores = new ScoreForSorting[mLabelsCount];
        for (int i = 0; i < mLabelsCount; ++i) {
            sortedAverageScores[i] = new ScoreForSorting(averageScores[i], i);
        }
        Arrays.sort(sortedAverageScores);

        // See if the latest top score is enough to trigger a detection.
        final int currentTopIndex = sortedAverageScores[0].mIndex;
        final String currentTopLabel = mLabels.get(currentTopIndex);
        final float currentTopScore = sortedAverageScores[0].mScore;
        // If we've recently had another label trigger, assume one that occurs too
        // soon afterwards is a bad result.
        long timeSinceLastTop;
        if (mPreviousTopLabel.equals("_silence_") ||
                (mPreviousTopLabelTime == Long.MIN_VALUE)) {
            timeSinceLastTop = Long.MAX_VALUE;
        } else {
            timeSinceLastTop = currentTimeMS - mPreviousTopLabelTime;
        }
        boolean isNewCommand;
        if ((currentTopScore > mDetectionThreshold) &&
                (!currentTopLabel.equals(mPreviousTopLabel)) &&
                (timeSinceLastTop > mSuppressionMS)) {
            mPreviousTopLabel = currentTopLabel;
            mPreviousTopLabelTime = currentTimeMS;
            mPreviousTopLabelScore = currentTopScore;
            isNewCommand = true;
        } else {
            isNewCommand = false;
        }
        return new RecognitionResult(currentTopLabel, currentTopScore, isNewCommand);
    }

}
