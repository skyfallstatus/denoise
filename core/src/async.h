// SPDX-License-Identifier: MIT
// Process-wide count of the core's asynchronous work: Metal completion
// handlers of renders that did not wait, the noise estimates queued after
// them, and GPU memory queued for release. od_wait_idle() waits for it to
// reach zero (a host's global teardown, before its plug-in is unloaded).
// Defined in session.mm.
#ifndef OD_ASYNC_H
#define OD_ASYNC_H

namespace od {

void processAsyncBegin();
void processAsyncEnd();
int processAsyncPending();

}  // namespace od

#endif
