# Mantra Algorithm

Mantra detects melodic similarity using **interval fingerprinting**.

The core idea is that melodies can be represented by **interval relationships between notes**, which remain stable under transposition.

---

# Pitch Representation

Each MIDI note is represented by its pitch value.

Example melody:

60 62 64 65

These numbers correspond to MIDI pitches.

---

# Interval Transformation

The melody is converted to intervals.

Example:

60 62 64 65

Intervals:

+2 +2 +1

This transformation makes the representation **key invariant**.

---

# Fingerprint Construction

The interval sequence is transformed into a fingerprint.

Example fingerprint:

[2, 2, 1]

Fingerprints allow efficient similarity comparison.

---

# Similarity Computation

Two melodies are compared by analyzing overlap between interval sequences.

Possible metrics include:

- sequence overlap
- edit distance
- structural similarity

---

# Example

Melody A:

C D E G

Melody B:

F G A C

Pitch values:

60 62 64 67  
65 67 69 72

Intervals:

+2 +2 +3  
+2 +2 +3

The interval fingerprints are identical.

Result:

Similarity score = 1.0

---

# Regression Testing

Mantra uses regression tests to ensure algorithm stability.

Test categories include:

- identical melodies
- transposed melodies
- modified melodies

---

# Limitations

Current algorithm focuses on **monophonic melodies**.

Future improvements:

- polyphonic analysis
- rhythmic similarity
- harmonic similarity

---

# Future Extensions

The algorithm can be expanded with:

- machine learning embeddings
- audio fingerprinting
- large-scale vector search