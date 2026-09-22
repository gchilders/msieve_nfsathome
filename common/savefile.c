/*--------------------------------------------------------------------
This source distribution is placed in the public domain by its author,
Jason Papadopoulos. You may use it for any purpose, free of charge,
without having to notify anyone. I disclaim any responsibility for any
errors.

Optionally, please be nice and tell me if you find this source to be
useful. Again optionally, if you add to the functionality present here
please consider making those additions public too, so that others may 
benefit from your work.	

$Id$
--------------------------------------------------------------------*/

#include <common.h>

#define SAVEFILE_STAGE_BUF (4 * 1024 * 1024)

/* we need a generic interface for reading and writing lines
   of data to the savefile while a factorization is in progress.
   This is necessary for two reasons: first, early msieve 
   versions would sometimes clobber their savefiles, and some
   users have several machines all write to the same savefile
   in a network directory. When output is manually buffered and 
   then explicitly flushed after writing to disk, most of the
   relations in the savefile will survive under these circumstances.

   The other reason is Windows-specific. Microsoft's C runtime 
   library has a bug that causes writes to files more than 4GB in
   size to fail. Thus, to deal with really large savefiles we have
   to call Win32 API functions directly, and these do not have
   any stdio-like stream functionality. Hence we need a homebrew
   implementation of some of stdio.h for the rest of the library
   to use */

#define SAVEFILE_BUF_SIZE 65536

/*--------------------------------------------------------------------*/
static const char *savefile_basename(const char *path) {

	const char *p = path;
	const char *slash = path;

	for (; *p; p++) {
		if (*p == '/' || *p == '\\')
			slash = p + 1;
	}
	return slash;
}

void get_filter_tmp_name(msieve_obj *obj, char *buf,
			size_t buf_len, const char *suffix) {

	int len;

	if (obj->scratch_dir == NULL) {
		len = snprintf(buf, buf_len, "%s%s", obj->savefile.name,
				suffix);
	}
	else {
		len = snprintf(buf, buf_len, "%s/%s%s", obj->scratch_dir,
				savefile_basename(obj->savefile.name), suffix);
	}

	/* truncation is not survivable here: several of these names differ
	   only in their last characters, so a truncated .lp0 can come out
	   equal to .lp and the rename that installs one over the other
	   would quietly destroy the file */

	if (len < 0 || (size_t)len >= buf_len) {
		printf("error: filtering path for '%s' does not fit in %u "
			"bytes\n", suffix, (uint32)buf_len);
		exit(-1);
	}
}

/*--------------------------------------------------------------------*/
/* The matrix is written once by the build, read back, reduced, and only
   then written out in the form the linear algebra consumes. The first two
   of those touch a file the size of the finished matrix -- 15.7 GB on a
   C223 -- and it is thrown away afterwards, so with a scratch directory
   configured it belongs there. The reduced matrix still goes beside the
   savefile, where the linear algebra and any restart expect it. */

void get_matrix_work_name(msieve_obj *obj, char *buf, size_t buf_len) {

	get_filter_tmp_name(obj, buf, buf_len, ".mat");
}

/*--------------------------------------------------------------------*/
/* Filtering reads the savefile three times and re-reads its own
   intermediates several more times. When those sit on a network
   filesystem that traffic dominates the run, so given a scratch
   directory we copy the savefile there once, decompressing on the way,
   and read from the copy thereafter.

   The original stays authoritative: appends still go to it, and the
   outputs that outlive filtering (.cyc, .rmap) are still written beside
   it. Only reads are redirected. */

uint32 savefile_stage(msieve_obj *obj) {

	char staged[256];
	char src[256];
	char name_gz[256];
	gzFile in;
	FILE *out;
	char *buf;
	int n;
	uint64 total = 0;
	time_t start = time(NULL);
#if defined(WIN32) || defined(_WIN64)
	struct _stati64 dummy;
#else
	struct stat dummy;
#endif

	if (obj->scratch_dir == NULL || obj->savefile.staged_name != NULL)
		return 0;

	get_filter_tmp_name(obj, staged, sizeof(staged), "");
	if (strcmp(staged, obj->savefile.name) == 0)
		return 0;

	if (snprintf(name_gz, sizeof(name_gz), "%s.gz",
			obj->savefile.name) >= (int)sizeof(name_gz)) {
		logprintf(obj, "error: savefile path too long to stage\n");
		exit(-1);
	}
#if defined(WIN32) || defined(_WIN64)
	if (_stati64(obj->savefile.name, &dummy) == 0)
#else
	if (stat(obj->savefile.name, &dummy) == 0)
#endif
		snprintf(src, sizeof(src), "%s", obj->savefile.name);
	else
		snprintf(src, sizeof(src), "%s", name_gz);

	/* Failing here cannot be shrugged off: the filtering intermediates
	   are routed to the same directory whether or not the savefile was
	   staged, so carrying on would only fail later with a message about
	   some unrelated file. Say which directory is at fault instead. */

	in = gzopen(src, "rb");
	if (in == NULL) {
		logprintf(obj, "error: cannot read '%s' to stage it\n", src);
		exit(-1);
	}
	out = fopen(staged, "wb");
	if (out == NULL) {
		logprintf(obj, "error: cannot write to scratch directory '%s'\n",
				obj->scratch_dir);
		gzclose(in);
		exit(-1);
	}

	buf = (char *)xmalloc(SAVEFILE_STAGE_BUF);
	gzbuffer(in, 1 << 20);
	while ((n = gzread(in, buf, SAVEFILE_STAGE_BUF)) > 0) {
		if (fwrite(buf, 1, (size_t)n, out) != (size_t)n) {
			logprintf(obj, "error: write failed staging savefile\n");
			free(buf); fclose(out); gzclose(in);
			remove(staged);
			exit(-1);
		}
		total += (uint64)n;
	}

	/* gzread reports both end of stream and failure by returning a
	   value that is not positive. Treating a failure as the end would
	   leave a truncated copy that every later pass reads as the whole
	   dataset, quietly filtering a subset of the relations. */

	if (n < 0) {
		int zerr = 0;
		const char *msg = gzerror(in, &zerr);

		logprintf(obj, "error: read failed staging '%s': %s\n",
				src, msg ? msg : "unknown");
		free(buf); fclose(out); gzclose(in);
		remove(staged);
		exit(-1);
	}
	free(buf);
	gzclose(in);
	if (fclose(out) != 0) {
		logprintf(obj, "error: cannot finalize staged savefile\n");
		remove(staged);
		exit(-1);
	}

	obj->savefile.staged_name = strdup(staged);
	logprintf(obj, "staged savefile to %s (%.1f MB in %u sec)\n",
			staged, (double)total / 1048576,
			(uint32)(time(NULL) - start));
	return 1;
}

/*--------------------------------------------------------------------*/
/* remove the staged savefile and any filtering intermediates left on
   scratch. Safe to call when nothing was staged. */

void savefile_unstage_tmp(msieve_obj *obj) {

	static const char *suffixes[] = { ".d", ".br", ".hc", ".lp", ".lp0" };
	char buf[256];
	size_t i;

	if (obj->scratch_dir == NULL)
		return;

	for (i = 0; i < sizeof(suffixes) / sizeof(suffixes[0]); i++) {
		get_filter_tmp_name(obj, buf, sizeof(buf), suffixes[i]);
		remove(buf);
	}
}

/*--------------------------------------------------------------------*/
void savefile_unstage(msieve_obj *obj) {

	savefile_unstage_tmp(obj);

	if (obj->scratch_dir != NULL && obj->savefile.staged_name != NULL) {
		remove(obj->savefile.staged_name);
		free(obj->savefile.staged_name);
		obj->savefile.staged_name = NULL;
	}
}

/*--------------------------------------------------------------------*/
void savefile_init(savefile_t *s, char *savefile_name) {
	
	memset(s, 0, sizeof(savefile_t));

	s->name = MSIEVE_DEFAULT_SAVEFILE;
	if (savefile_name)
		s->name = savefile_name;
	
	s->buf = (char *)xmalloc((size_t)SAVEFILE_BUF_SIZE);
}

/*--------------------------------------------------------------------*/
void savefile_free(savefile_t *s) {
	
	free(s->buf);
	memset(s, 0, sizeof(savefile_t));
}

/*--------------------------------------------------------------------*/
void savefile_open(savefile_t *s, uint32 flags) {

	char *nm = s->name;

	/* a scratch copy of the savefile only serves reads. Appends keep
	   going to the original, which stays the authoritative file that
	   .cyc relation indices refer to */

	if (s->staged_name != NULL && (flags & SAVEFILE_READ) &&
			!(flags & (SAVEFILE_WRITE | SAVEFILE_APPEND)))
		nm = s->staged_name;

#if defined(NO_ZLIB) && (defined(WIN32) || defined(_WIN64))
	DWORD access_arg, open_arg;

	if (flags & SAVEFILE_READ)
		access_arg = GENERIC_READ;
	else
		access_arg = GENERIC_WRITE;

	if (flags & SAVEFILE_READ)
		open_arg = OPEN_EXISTING;
	else if (flags & SAVEFILE_APPEND)
		open_arg = OPEN_ALWAYS;
	else
		open_arg = CREATE_ALWAYS;

	s->file_handle = CreateFile(nm, 
					access_arg,
					FILE_SHARE_READ |
					FILE_SHARE_WRITE, NULL,
					open_arg,
					FILE_FLAG_SEQUENTIAL_SCAN,
					NULL);

	if (s->file_handle == INVALID_HANDLE_VALUE) {
		printf("error: cannot open '%s'", nm);
		exit(-1);
	}
	if (flags & SAVEFILE_APPEND) {
		LARGE_INTEGER fileptr;
		fileptr.QuadPart = 0;
		SetFilePointerEx(s->file_handle, 
				fileptr, NULL, FILE_END);
	}
	s->read_size = 0;
	s->eof = 0;

#else
	char *open_string;
#ifndef NO_ZLIB
	char name_gz[256];
	#if defined(WIN32) || defined(_WIN64)
	struct _stati64 dummy;
	#else
	struct stat dummy;
	#endif
#endif

	if (flags & SAVEFILE_APPEND)
		open_string = "a";
	else if ((flags & SAVEFILE_READ) && (flags & SAVEFILE_WRITE))
		open_string = "r+w";
	else if (flags & SAVEFILE_READ)
		open_string = "r";
	else
		open_string = "w";

	s->is_a_FILE = s->isCompressed = 0;

#ifndef NO_ZLIB
	sprintf(name_gz, "%s.gz", nm);
	#if defined(WIN32) || defined(_WIN64)
	if (_stati64(name_gz, &dummy) == 0) {
		if (_stati64(nm, &dummy) == 0) {
	#else
	if (stat(name_gz, &dummy) == 0) {
		if (stat(nm, &dummy) == 0) {
	#endif
			printf("error: both '%s' and '%s' exist. "
			       "Remove the wrong one and restart\n",
				nm, name_gz);
			exit(-1);
		}
		s->isCompressed = 1;
		s->fp = gzopen(name_gz, open_string);
		if (s->fp == NULL) {
			printf("error: cannot open '%s'\n", name_gz);
			exit(-1);
		}
		/* fprintf(stderr, "using compressed '%s'\n", name_gz); */
	} else if (flags & SAVEFILE_APPEND) {
		/* Unfortunately, append is not intuitive in zlib */
		/* Note: the .dat file may be a compressed file   */
		/*       we are using UNIX philosophy here:       */
		/*       it is the content, not filename, that matters */
		uint8 header[4];
		FILE *fp;
		int n;

		if((fp = fopen(nm, "r"))) {
			if((n = fread(header, sizeof(uint8), 3, fp)) && 
		   	   (n != 3 || header[0]!=31 || header[1]!=139 || header[2]!=8))
				s->is_a_FILE = 1; 
			/* exists, non-empty and not gzipped,
			   so we will fopen a FILE to append plainly */
			fclose(fp);
		}
		if (s->is_a_FILE) {
			s->fp = (gzFile)fopen(nm, "a");
		} else {
			s->fp = gzopen(nm, "a");
			s->isCompressed = 1;
		}
	} else
#endif
	{
		s->fp = gzopen(nm, open_string);
	}
	if (s->fp == NULL) {
		printf("error: cannot open '%s'\n", nm);
		exit(-1);
	}

	/* zlib defaults to an 8KB buffer, and refills it a great many
	   times over a savefile of tens of gigabytes */

	if (!s->is_a_FILE)
		gzbuffer((gzFile)s->fp, 1 << 20);
#endif

	s->buf_off = 0;
	s->buf[0] = 0;
}

/*--------------------------------------------------------------------*/
void savefile_close(savefile_t *s) {
	
#if defined(NO_ZLIB) && (defined(WIN32) || defined(_WIN64))
	CloseHandle(s->file_handle);
	s->file_handle = INVALID_HANDLE_VALUE;
#else
	s->is_a_FILE ? fclose((FILE *)s->fp) : gzclose(s->fp);
	s->fp = NULL;
#endif
}

/*--------------------------------------------------------------------*/
uint32 savefile_eof(savefile_t *s) {
	
#if defined(NO_ZLIB) && (defined(WIN32) || defined(_WIN64))
	return (s->buf_off == s->read_size && s->eof);
#else
	return (s->is_a_FILE ? feof((FILE *)s->fp) : gzeof(s->fp));
#endif
}

/*--------------------------------------------------------------------*/
uint32 savefile_exists(savefile_t *s) {
	
#if defined(WIN32) || defined(_WIN64)
	struct _stati64 dummy;
	return (_stati64(s->name, &dummy) == 0);
#else
	struct stat dummy;
	return (stat(s->name, &dummy) == 0);
#endif
}

/*--------------------------------------------------------------------*/
void savefile_read_line(char *buf, size_t max_len, savefile_t *s) {

#if defined(NO_ZLIB) && (defined(WIN32) || defined(_WIN64))
	size_t i, j;
	char *sbuf = s->buf;

	for (i = s->buf_off, j = 0; i < s->read_size && 
				j < max_len - 1; i++, j++) { /* read bytes */
		buf[j] = sbuf[i];
		if (buf[j] == '\n' || buf[j] == '\r') {
			buf[j+1] = 0;
			s->buf_off = i + 1;
			return;
		}
	}
	s->buf_off = i;
	if (i == s->read_size && !s->eof) {	/* sbuf ran out? */
		DWORD num_read;
		ReadFile(s->file_handle, sbuf, 
				SAVEFILE_BUF_SIZE, 
				&num_read, NULL);
		s->read_size = num_read;
		s->buf_off = 0;

		/* set EOF only if previous lines have exhausted sbuf
		   and there are no more bytes in the file */

		if (num_read == 0)
			s->eof = 1;
	}
	for (i = s->buf_off; i < s->read_size && 
				j < max_len - 1; i++, j++) { /* read more */
		buf[j] = sbuf[i];
		if (buf[j] == '\n' || buf[j] == '\r') {
			i++; j++;
			break;
		}
	}
	buf[j] = 0;
	s->buf_off = i;
#else
	gzgets(s->fp, buf, (int)max_len);
#endif
}

/*--------------------------------------------------------------------*/
void savefile_write_line(savefile_t *s, char *buf) {

	if (s->buf_off + strlen(buf) + 1 >= SAVEFILE_BUF_SIZE)
		savefile_flush(s);

	s->buf_off += sprintf(s->buf + s->buf_off, "%s", buf);
}

/*--------------------------------------------------------------------*/
void savefile_flush(savefile_t *s) {

#if defined(NO_ZLIB) && (defined(WIN32) || defined(_WIN64))
	if (s->buf_off) {
		DWORD num_write; /* required because of NULL arg below */
		WriteFile(s->file_handle, s->buf, 
				s->buf_off, &num_write, NULL);
	}
	FlushFileBuffers(s->file_handle);
#else
	if (s->is_a_FILE) {
		fprintf((FILE *)s->fp, "%s", s->buf);
		fflush((FILE *)s->fp);
	} else {
		gzputs(s->fp, s->buf);
	}
#endif

	s->buf_off = 0;
	s->buf[0] = 0;
}

/*--------------------------------------------------------------------*/
void savefile_rewind(savefile_t *s) {

#if defined(NO_ZLIB) && (defined(WIN32) || defined(_WIN64))
	LARGE_INTEGER fileptr;
	fileptr.QuadPart = 0;
	SetFilePointerEx(s->file_handle, fileptr, NULL, FILE_BEGIN);
	s->read_size = 0;   /* invalidate buffered data */
	s->buf_off = 0;
	s->eof = 0;
#else
	s->is_a_FILE ? rewind((FILE *)s->fp) : gzrewind(s->fp);
#endif
}

